"""Day 3 #3 — single-turn Gemini tool calling over the EIA generation fetch.

Exposes ``EIAClient.fetch_generation`` to Gemini as a ``get_generation`` tool.
Gemini decides to call it, we execute the fetch locally, hand back a compact
summary, and Gemini writes the final natural-language answer. One round trip of
tool use only — no multi-step ReAct loop (that's Day 4).

Run:
    uv run python -m gridsense.llm.tool_call "gas vs solar mix in CAISO, first week of June 2025?"

Diagnostic lines (the tool call and its result) are printed to stderr so stdout
carries only the final answer.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date
from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from gridsense.eia.client import EIAClient

MODEL = "gemini-2.5-flash"

# If the model emits a colloquial ISO name instead of the EIA respondent code,
# normalize it (EIA uses CISO for CAISO, ERCO for ERCOT, etc.).
_NAME_TO_CODE = {
    "CAISO": "CISO",
    "ERCOT": "ERCO",
    "NYISO": "NYIS",
    "ISO-NE": "ISNE",
    "ISONE": "ISNE",
    "SPP": "SWPP",
}

GET_GENERATION = types.FunctionDeclaration(
    name="get_generation",
    description=(
        "Fetch hourly net electricity generation by fuel type for a US ISO/RTO "
        "over an inclusive date range. Returns generation totals (MWh) per fuel type."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "region": {
                "type": "string",
                "description": (
                    "EIA respondent code. Use CISO for the California ISO (CAISO), "
                    "ERCO for ERCOT, PJM, NYIS for NYISO, ISNE for ISO-NE, MISO, SWPP for SPP."
                ),
            },
            "start": {"type": "string", "description": "Start date, inclusive, format YYYY-MM-DD."},
            "end": {"type": "string", "description": "End date, inclusive, format YYYY-MM-DD."},
        },
        "required": ["region", "start", "end"],
    },
)


async def _fetch(region: str, start: date, end: date) -> pd.DataFrame:
    eia_key = os.environ.get("EIA_API_KEY")
    if not eia_key:
        raise SystemExit("EIA_API_KEY not set (needed to execute get_generation).")
    async with EIAClient(eia_key) as client:
        return await client.fetch_generation(region, start, end)


def run_get_generation(args: dict[str, object]) -> dict[str, object]:
    """Execute the tool locally and return a compact, JSON-serializable summary."""
    region_raw = str(args["region"]).upper()
    region = _NAME_TO_CODE.get(region_raw, region_raw)
    start = date.fromisoformat(str(args["start"]))
    end = date.fromisoformat(str(args["end"]))

    df = asyncio.run(_fetch(region, start, end))
    if df.empty:
        return {"region": region, "rows": 0, "note": "no data returned for this range"}

    by_fuel = df.groupby("type_name", observed=True)["value"].sum().sort_values(ascending=False)
    return {
        "region": region,
        "period_start": str(df["period"].min()),
        "period_end": str(df["period"].max()),
        "rows": int(len(df)),
        "generation_mwh_by_fuel": {str(k): round(float(v), 1) for k, v in by_fuel.items()},
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m gridsense.llm.tool_call "<question>"')
    question = sys.argv[1]

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set (add it to .env or export it in your shell).")

    client = genai.Client(api_key=api_key)
    tool = types.Tool(function_declarations=[GET_GENERATION])
    config = types.GenerateContentConfig(
        tools=[tool],
        # Drive the tool loop by hand so the protocol is visible (Day 4 ReAct builds on it).
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Turn 1: send the question, let Gemini decide whether to call the tool.
    user_content = types.Content(role="user", parts=[types.Part(text=question)])
    first = client.models.generate_content(
        model=MODEL, contents=cast(Any, [user_content]), config=config
    )

    calls = first.function_calls
    if not calls:
        # Gemini answered directly without using the tool.
        print(first.text)
        return

    call = calls[0]
    assert call.name is not None  # a function call always names its function
    args = dict(call.args or {})
    print(f"[tool] Gemini requested {call.name}({args})", file=sys.stderr)

    result = run_get_generation(args)
    print(f"[tool] local result: {result}", file=sys.stderr)

    if not first.candidates or first.candidates[0].content is None:
        raise SystemExit("Unexpected: Gemini returned no candidate content.")
    model_content = first.candidates[0].content  # the turn containing the function call

    # Turn 2: hand the result back as a `tool` message; Gemini writes the answer.
    tool_content = types.Content(
        role="tool",
        parts=[types.Part.from_function_response(name=call.name, response={"result": result})],
    )
    final = client.models.generate_content(
        model=MODEL,
        contents=cast(Any, [user_content, model_content, tool_content]),
        config=config,
    )
    print(final.text)


if __name__ == "__main__":
    main()
