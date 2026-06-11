#!/usr/bin/env python3
"""Launch the Book Writing Agent web UI.

Usage:
    python run_ui.py
    python run_ui.py --port 3000

Open http://localhost:3000 in your browser.

Environment:
    OLLAMA_BASE_URL  - Ollama API (default http://localhost:11434, use SSH tunnel to server)
    FORGE_API_URL    - Forge/SD API (default http://localhost:7860)
"""

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Book Writing Agent UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 3000)))
    args = parser.parse_args()

    print(f"Book Writing UI: http://{args.host}:{args.port}")
    print(f"Ollama (text):  {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"Forge (image):  {os.environ.get('FORGE_API_URL', 'http://localhost:7860')}")

    uvicorn.run("app.ui_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
