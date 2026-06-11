"""Background job runner for book / picture-book pipelines."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from slugify import slugify

from app.pipeline_config import (
    build_picture_cli_args,
    build_text_cli_args,
    normalize_pipeline,
    normalize_picture_agents,
    pipeline_from_toc,
)
from app.toc_io import save_toc_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("book-agent")


def _kill_process(proc: subprocess.Popen, timeout: float = 5.0) -> int | None:
    """Terminate a subprocess; escalate to kill (incl. child tree on Windows)."""
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except Exception:
            logger.exception("taskkill failed for pid %s", proc.pid)
            proc.kill()
    else:
        proc.kill()
    try:
        return proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        logger.warning("Process pid %s did not exit after kill", proc.pid)
        return None


def _default_output_dir(toc_path: Path, book_type: str) -> str:
    data = json.loads(toc_path.read_text(encoding="utf-8"))
    slug = slugify(data.get("title", "untitled"), max_length=60)
    base = "picture-book" if book_type == "picture" else "book"
    return str(PROJECT_ROOT / base / slug)


@dataclass
class Job:
    id: str
    book_type: Literal["text", "picture"]
    toc_path: str
    output_dir: str
    text_model: str
    image_backend: str
    image_model: str
    status: Literal["running", "completed", "failed", "stopped"] = "running"
    started_at: str = ""
    finished_at: str = ""
    return_code: int | None = None
    log_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = field(default=None, repr=False)

    def _read_log_file(self) -> list[str]:
        name = "picture-book.log" if self.book_type == "picture" else "book-writer.log"
        path = Path(self.output_dir) / name
        if not path.exists():
            return []
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

    def _log_tail(self, n: int = 200) -> list[str]:
        file_lines = self._read_log_file()
        if file_lines:
            return file_lines[-n:]
        return self.log_lines[-n:]

    def to_dict(self) -> dict:
        progress = self._read_progress()
        return {
            "id": self.id,
            "book_type": self.book_type,
            "toc_path": self.toc_path,
            "output_dir": self.output_dir,
            "text_model": self.text_model,
            "image_backend": self.image_backend,
            "image_model": self.image_model,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "progress": progress,
            "log_tail": self._log_tail(200),
            "log_file": str(
                Path(self.output_dir)
                / ("picture-book.log" if self.book_type == "picture" else "book-writer.log")
            ),
        }

    def _read_progress(self) -> dict:
        path = Path(self.output_dir) / ".progress.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"completed": [], "failed": {}, "in_progress": None}


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in reversed(list(self._jobs.values()))]

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def start_job(
        self,
        *,
        book_type: str,
        toc_path: str,
        text_model: str,
        image_backend: str = "automatic1111",
        image_model: str = "",
        output_dir: str = "",
        resume: bool = False,
        rewrite_all: bool = False,
        rewrite: list[int] | None = None,
        no_push: bool = True,
        pipeline: dict | None = None,
    ) -> dict:
        toc = Path(toc_path)
        if not toc.is_absolute():
            toc = PROJECT_ROOT / toc
        if not toc.exists():
            raise FileNotFoundError(f"TOC not found: {toc}")

        toc_data = json.loads(toc.read_text(encoding="utf-8"))
        file_book_type = (
            "picture" if toc_data.get("type") == "picture_book" else "text"
        )
        if file_book_type != book_type:
            logger.warning(
                "UI book_type=%r does not match TOC (%s); using %r",
                book_type,
                toc.name,
                file_book_type,
            )
            book_type = file_book_type

        if pipeline:
            pipe = normalize_pipeline(book_type, pipeline)
        else:
            pipe = pipeline_from_toc(toc_data)

        if book_type == "picture":
            pipe["agents"] = normalize_picture_agents(pipe.get("agents"))
            try:
                saved = save_toc_pipeline(PROJECT_ROOT, str(toc), pipe)
                pipe = saved["pipeline"]
                logger.info("TOC pipeline saved: agents=%s", pipe.get("agents"))
            except Exception:
                logger.exception("Failed to persist normalized pipeline to TOC")

        if not output_dir:
            output_dir = _default_output_dir(toc, book_type)

        stopped = self.stop_jobs_for_output(output_dir)
        if stopped:
            logger.info("Stopped %d previous job(s) for %s", stopped, output_dir)

        job_id = str(uuid.uuid4())[:8]
        if book_type == "picture":
            script = PROJECT_ROOT / "run_picture_book.py"
            cmd = [
                sys.executable, str(script),
                "--toc", str(toc),
                "--model", text_model,
                "--image-backend", image_backend,
                "--no-push" if no_push else "",
            ]
            if image_model:
                cmd.extend(["--image-model", image_model])
            if output_dir:
                cmd.extend(["--output-dir", output_dir])
            if resume:
                cmd.append("--resume")
            if rewrite_all:
                cmd.append("--rewrite-all")
            elif rewrite:
                cmd.extend(["--rewrite", *[str(n) for n in rewrite]])
            cli_pipe_args = build_picture_cli_args(pipe)
            cmd.extend(cli_pipe_args)
            logger.info(
                "Picture pipeline agents: %s",
                pipe.get("agents"),
            )
            logger.info("CLI args: %s", " ".join(cli_pipe_args))
        else:
            script = PROJECT_ROOT / "run_book.py"
            cmd = [
                sys.executable, str(script),
                "--toc", str(toc),
                "--model", text_model,
                "--no-push" if no_push else "",
            ]
            if output_dir:
                cmd.extend(["--output-dir", output_dir])
            if resume:
                cmd.append("--resume")
            if rewrite_all:
                cmd.append("--rewrite-all")
            elif rewrite:
                cmd.extend(["--rewrite", *[str(n) for n in rewrite]])
            cmd.extend(build_text_cli_args(pipe))

        cmd = [c for c in cmd if c]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        job = Job(
            id=job_id,
            book_type=book_type,  # type: ignore[arg-type]
            toc_path=str(toc),
            output_dir=output_dir,
            text_model=text_model,
            image_backend=image_backend,
            image_model=image_model,
            started_at=datetime.now(timezone.utc).isoformat(),
            process=proc,
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._watch, args=(job_id,), daemon=True)
        thread.start()
        return job.to_dict()

    def _watch(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job or not job.process:
            return

        assert job.process.stdout
        for line in job.process.stdout:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].log_lines.append(line.rstrip())
                    if len(self._jobs[job_id].log_lines) > 2000:
                        self._jobs[job_id].log_lines = self._jobs[job_id].log_lines[-2000:]

        rc = job.process.wait()
        with self._lock:
            if job_id in self._jobs:
                j = self._jobs[job_id]
                if j.status != "running":
                    return
                j.return_code = rc
                j.status = "completed" if rc == 0 else "failed"
                j.finished_at = datetime.now(timezone.utc).isoformat()

    def stop_jobs_for_output(self, output_dir: str) -> int:
        """Stop running jobs writing to the same output folder (unlocks log files on Windows)."""
        target = Path(output_dir).resolve()
        to_stop: list[tuple[str, subprocess.Popen]] = []
        with self._lock:
            for job in self._jobs.values():
                if job.status != "running":
                    continue
                if Path(job.output_dir).resolve() != target:
                    continue
                if not job.process:
                    continue
                job.status = "stopped"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                to_stop.append((job.id, job.process))
        for job_id, proc in to_stop:
            rc = _kill_process(proc)
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.return_code = rc if rc is not None else -1
        return len(to_stop)

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != "running" or not job.process:
                return False
            proc = job.process
            job.status = "stopped"
            job.finished_at = datetime.now(timezone.utc).isoformat()
        rc = _kill_process(proc)
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.return_code = rc if rc is not None else -1
        logger.info("Job %s stopped (return_code=%s)", job_id, rc)
        return True


job_manager = JobManager()
