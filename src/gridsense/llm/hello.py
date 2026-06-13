"""Day 3 #1 — minimal Gemini smoke test.

Verifies the google-genai SDK works end-to-end: read the key, call the model,
print the answer. Run: ``uv run python -m gridsense.llm.hello``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-2.5-flash"


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set (add it to .env or export it in your shell).")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents="What is 2+2?")
    print(response.text)


if __name__ == "__main__":
    main()
