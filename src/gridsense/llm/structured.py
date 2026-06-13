"""Day 3 #2 — Gemini structured output validated by pydantic.

Asks Gemini to extract a typed summary from a free-text fact, using native JSON
mode with a pydantic ``response_schema``. The SDK returns a validated model
instance on ``response.parsed``. Run: ``uv run python -m gridsense.llm.structured``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

MODEL = "gemini-2.5-flash"

SAMPLE_FACT = (
    "On June 1, 2025, the California ISO (CAISO) grid leaned heavily on solar "
    "during midday hours. Over the full day, solar was the single largest source "
    "at about 38% of net generation, ahead of natural gas at roughly 30%, with "
    "imports, wind, and hydro making up most of the rest."
)


class EnergyMixSummary(BaseModel):
    """Structured summary of a region's generation mix."""

    region: str = Field(description="ISO / region name, e.g. CAISO")
    top_fuel: str = Field(description="Largest single generation fuel type")
    top_fuel_pct: float = Field(description="Top fuel's share of net generation, as a percent")
    summary_text: str = Field(description="One-sentence plain-language summary of the mix")


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set (add it to .env or export it in your shell).")

    client = genai.Client(api_key=api_key)
    prompt = f"Extract a structured energy generation summary from this fact:\n\n{SAMPLE_FACT}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EnergyMixSummary,
        ),
    )

    # The SDK parses the JSON into our pydantic model and exposes it on .parsed.
    summary = response.parsed
    print(summary)


if __name__ == "__main__":
    main()
