"""Day 4 Phase 1 — a manual ReAct loop over two EIA tools, using the Gemini SDK directly.

No agent framework. This is the core ReAct iteration spelled out by hand:

    seed the conversation with the user's question
    loop (up to MAX_ITERATIONS):
        send the WHOLE running conversation to Gemini
        if it returns text instead of a tool call -> that's the answer, stop
        otherwise: append the model's tool-call turn, execute every call,
                   append the results as one tool turn, and loop again

The API is stateless, so we resend the full ``contents`` history each turn and
must append both the model's call turn and our tool-result turn in order.

Run:
    uv run python -m gridsense.agents.react_manual "<question>"

Tool-call tracing is printed to stderr; only the final answer goes to stdout.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from gridsense.eia.client import EIAClient

MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 5

# Normalize colloquial ISO names to EIA respondent codes if the model emits them.
_NAME_TO_CODE = {
    "CAISO": "CISO",
    "ERCOT": "ERCO",
    "NYISO": "NYIS",
    "ISO-NE": "ISNE",
    "ISONE": "ISNE",
    "SPP": "SWPP",
}

_REGION_DESC = (
    "EIA respondent code. CISO=California ISO (CAISO), ERCO=ERCOT, PJM, "
    "NYIS=NYISO, ISNE=ISO-NE, MISO, SWPP=SPP."
)
_DATE_PARAMS = {
    "start": {"type": "string", "description": "Start date, inclusive, YYYY-MM-DD."},
    "end": {"type": "string", "description": "End date, inclusive, YYYY-MM-DD."},
}

GET_GENERATION = types.FunctionDeclaration(
    name="get_generation",
    description=(
        "Fetch net electricity generation by fuel type for a US ISO/RTO over an "
        "inclusive date range. Returns total MWh per fuel type (the generation mix)."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {"region": {"type": "string", "description": _REGION_DESC}, **_DATE_PARAMS},
        "required": ["region", "start", "end"],
    },
)

GET_DEMAND_AND_INTERCHANGE = types.FunctionDeclaration(
    name="get_demand_and_interchange",
    description=(
        "Fetch demand, net generation, and total interchange for a US ISO/RTO over "
        "an inclusive date range. Use this to decide whether a region was a net "
        "importer or exporter (positive total interchange = net exports, negative = "
        "net imports)."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {"region": {"type": "string", "description": _REGION_DESC}, **_DATE_PARAMS},
        "required": ["region", "start", "end"],
    },
)


def _eia_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise SystemExit("EIA_API_KEY not set (needed to execute the tools).")
    return key


def _parse_args(args: dict[str, Any]) -> tuple[str, date, date]:
    region = str(args["region"]).upper()
    region = _NAME_TO_CODE.get(region, region)
    return region, date.fromisoformat(str(args["start"])), date.fromisoformat(str(args["end"]))


async def _fetch_generation(region: str, start: date, end: date) -> pd.DataFrame:
    async with EIAClient(_eia_key()) as client:
        return await client.fetch_generation(region, start, end)


async def _fetch_region(region: str, start: date, end: date) -> pd.DataFrame:
    async with EIAClient(_eia_key()) as client:
        return await client.fetch_region_data(region, start, end)


async def summarize_generation(region: str, start: date, end: date) -> dict[str, Any]:
    """Async core: fetch generation and summarize MWh by fuel type."""
    df = await _fetch_generation(region, start, end)
    if df.empty:
        return {"region": region, "rows": 0, "note": "no data for this range"}
    by_fuel = df.groupby("type_name", observed=True)["value"].sum().sort_values(ascending=False)
    return {
        "region": region,
        "rows": int(len(df)),
        "generation_mwh_by_fuel": {str(k): round(float(v), 1) for k, v in by_fuel.items()},
    }


async def summarize_demand_and_interchange(region: str, start: date, end: date) -> dict[str, Any]:
    """Async core: fetch region data and summarize demand / net gen / interchange."""
    df = await _fetch_region(region, start, end)
    if df.empty:
        return {"region": region, "rows": 0, "note": "no data for this range"}
    by_type = df.groupby("type", observed=True)["value"].sum()
    labels = {"D": "demand_mwh", "NG": "net_generation_mwh", "TI": "total_interchange_mwh"}
    totals = {labels.get(str(k), str(k)): round(float(v), 1) for k, v in by_type.items()}
    return {
        "region": region,
        "rows": int(len(df)),
        "totals": totals,
        "convention": "total_interchange_mwh > 0 means net exports; < 0 means net imports",
    }


def run_get_generation(args: dict[str, Any]) -> dict[str, Any]:
    """Sync wrapper for the manual loop (no running event loop here)."""
    region, start, end = _parse_args(args)
    return asyncio.run(summarize_generation(region, start, end))


def run_get_demand_and_interchange(args: dict[str, Any]) -> dict[str, Any]:
    """Sync wrapper for the manual loop (no running event loop here)."""
    region, start, end = _parse_args(args)
    return asyncio.run(summarize_demand_and_interchange(region, start, end))


DISPATCH = {
    "get_generation": run_get_generation,
    "get_demand_and_interchange": run_get_demand_and_interchange,
}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m gridsense.agents.react_manual "<question>"')
    question = sys.argv[1]

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set (add it to .env).")

    client = genai.Client(api_key=api_key)
    tool = types.Tool(function_declarations=[GET_GENERATION, GET_DEMAND_AND_INTERCHANGE])
    config = types.GenerateContentConfig(
        tools=[tool],
        # Drive the loop by hand so we can see the protocol (no auto function calling).
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=question)])]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)

        calls = response.function_calls
        if not calls:
            # Stop condition: the model answered in prose instead of calling a tool.
            print(response.text)
            return

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            raise SystemExit("Unexpected: Gemini returned no candidate content.")
        contents.append(candidate.content)  # the model's tool-call turn

        tool_parts: list[types.Part] = []
        for call in calls:
            assert call.name is not None  # a function call always names its function
            args = dict(call.args or {})
            print(f"[tool] iter {iteration}: call {call.name}({args})", file=sys.stderr)
            executor = DISPATCH.get(call.name)
            result = executor(args) if executor else {"error": f"unknown tool {call.name}"}
            print(f"[tool] iter {iteration}: {call.name} -> {result}", file=sys.stderr)
            tool_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )

        contents.append(types.Content(role="tool", parts=tool_parts))  # all results, one turn

    print(f"[agent] stopped after {MAX_ITERATIONS} iterations without an answer.", file=sys.stderr)


if __name__ == "__main__":
    main()
