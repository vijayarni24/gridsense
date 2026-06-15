"""Day 4 Phase 2 — the same two-tool agent, ported to Google ADK.

Contrast with ``react_manual.py``. Here we write **neither** a FunctionDeclaration
**nor** a ReAct loop:

- Tools are plain Python functions. ADK derives each tool's schema from the
  function signature + docstring (so the docstring *is* the declaration).
- An ``Agent`` bundles the model, instruction, and tools.
- The ``Runner`` drives the observe -> call -> execute -> repeat loop, accumulates
  conversation history in a session service, and stops on the final response —
  all the bookkeeping we wrote by hand in Phase 1.

The underlying tool logic is imported from ``react_manual`` unchanged, so the
only thing that differs between the two files is the framework wiring.

Run:
    uv run python -m gridsense.agents.react_adk "<question>"

Tool tracing goes to stderr; only the final answer goes to stdout.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from gridsense.agents.react_manual import (
    _parse_args,
    summarize_demand_and_interchange,
    summarize_generation,
)

MODEL = "gemini-2.5-flash"
APP_NAME = "gridsense"
USER_ID = "local"
SESSION_ID = "day4-phase2"


async def get_generation(region: str, start: str, end: str) -> dict[str, Any]:
    """Get net electricity generation by fuel type for a US ISO/RTO (the generation mix).

    Args:
        region: EIA respondent code, e.g. CISO (California ISO / CAISO), ERCO (ERCOT),
            PJM, NYIS (NYISO), ISNE (ISO-NE), MISO, SWPP (SPP).
        start: Start date, inclusive, formatted YYYY-MM-DD.
        end: End date, inclusive, formatted YYYY-MM-DD.

    Returns:
        Total MWh per fuel type over the range.
    """
    reg, s, e = _parse_args({"region": region, "start": start, "end": end})
    return await summarize_generation(reg, s, e)


async def get_demand_and_interchange(region: str, start: str, end: str) -> dict[str, Any]:
    """Get demand, net generation, and total interchange for a US ISO/RTO.

    Use this to decide whether a region was a net importer or exporter: total
    interchange > 0 means net exports, < 0 means net imports.

    Args:
        region: EIA respondent code, e.g. CISO (California ISO / CAISO), ERCO (ERCOT),
            PJM, NYIS (NYISO), ISNE (ISO-NE), MISO, SWPP (SPP).
        start: Start date, inclusive, formatted YYYY-MM-DD.
        end: End date, inclusive, formatted YYYY-MM-DD.

    Returns:
        Total demand, net generation, and total interchange (MWh) over the range.
    """
    reg, s, e = _parse_args({"region": region, "start": start, "end": end})
    return await summarize_demand_and_interchange(reg, s, e)


agent = Agent(
    name="gridsense_analyst",
    model=MODEL,
    instruction=(
        "You are an energy grid analyst. Answer questions about US ISO/RTO electricity "
        "using the available tools, and ground every claim in the tool results."
    ),
    tools=[get_generation, get_demand_and_interchange],
)


def _trace(event: Event) -> None:
    """Mirror Phase 1's stderr trace by inspecting the event's function parts."""
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if part.function_call:
            args = dict(part.function_call.args or {})
            print(f"[tool] call {part.function_call.name}({args})", file=sys.stderr)
        if part.function_response:
            print(
                f"[tool] {part.function_response.name} -> {part.function_response.response}",
                file=sys.stderr,
            )


async def _run(question: str) -> str:
    session_service = InMemorySessionService()
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    message = types.Content(role="user", parts=[types.Part(text=question)])
    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=message
    ):
        _trace(event)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""
    return final_text


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m gridsense.agents.react_adk "<question>"')
    load_dotenv()
    print(asyncio.run(_run(sys.argv[1])))


if __name__ == "__main__":
    main()
