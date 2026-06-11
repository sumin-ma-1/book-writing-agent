import json
import os

from google.adk.cli.fast_api import get_fast_api_app

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=allow_origins,
    session_service_uri=None,
)
app.title = "book-writer"
app.description = "Book-writing agent API"


@app.get("/api/progress")
async def get_progress():
    output_dir = os.environ.get("BOOK_OUTPUT_DIR", "./book")
    progress_file = os.path.join(output_dir, ".progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return {"completed": [], "failed": {}, "in_progress": None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
