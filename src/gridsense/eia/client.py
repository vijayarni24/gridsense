"""Async client for EIA Open Data v2 — electricity generation by fuel type.

Design notes:
- Fully async so the data layer can later fan out across many regions/date
  chunks concurrently without a rewrite. The CLI bridges with ``asyncio.run``.
- Page 1 tells us the total row count; remaining pages (EIA caps at 5,000 rows
  each) are fetched concurrently behind a semaphore.
- Retries only on 429 + 5xx + transport errors. Other 4xx (bad key/params) are
  caller bugs and are surfaced immediately.
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import TracebackType

import httpx
import pandas as pd
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from gridsense.eia.models import EIAResponse, GenerationRecord

EIA_BASE_URL = "https://api.eia.gov/v2"
FUEL_TYPE_PATH = "/electricity/rto/fuel-type-data/data"
PAGE_LIMIT = 5000  # EIA hard cap on rows per request
MAX_ATTEMPTS = 5

_CATEGORICAL_COLS = ("respondent", "respondent_name", "fueltype", "type_name", "value_units")
_FRAME_COLS = ("period", *_CATEGORICAL_COLS, "value")


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient network errors and 429/5xx; never on other 4xx."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def _wait_strategy(retry_state: RetryCallState) -> float:
    """Honor a 429 ``Retry-After`` header when present; else exponential + jitter."""
    outcome = retry_state.outcome
    if outcome is not None:
        exc = outcome.exception()
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                return float(retry_after)
    return wait_exponential_jitter(initial=1.0, max=30.0)(retry_state)


class EIAClient:
    """Async EIA v2 client. Use as an async context manager."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = EIA_BASE_URL,
        timeout: float = 30.0,
        max_concurrency: int = 5,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._sem = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self) -> EIAClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    def _build_params(
        self, region: str, start: date, end: date, *, offset: int
    ) -> dict[str, str | int]:
        return {
            "api_key": self._api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "start": f"{start.isoformat()}T00",
            "end": f"{end.isoformat()}T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_LIMIT,
        }

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=_wait_strategy,
        reraise=True,
    )
    async def _get_page(self, params: dict[str, str | int]) -> EIAResponse:
        async with self._sem:
            resp = await self._client.get(FUEL_TYPE_PATH, params=params)
        resp.raise_for_status()
        payload = resp.json()
        return EIAResponse.model_validate(payload["response"])

    async def fetch_generation(self, region: str, start: date, end: date) -> pd.DataFrame:
        """Fetch hourly net generation by fuel type for ``region`` over [start, end]."""
        first = await self._get_page(self._build_params(region, start, end, offset=0))
        records: list[GenerationRecord] = list(first.data)

        if first.total > PAGE_LIMIT:
            offsets = range(PAGE_LIMIT, first.total, PAGE_LIMIT)
            pages = await asyncio.gather(
                *(self._get_page(self._build_params(region, start, end, offset=o)) for o in offsets)
            )
            for page in pages:
                records.extend(page.data)

        return self._to_dataframe(records)

    @staticmethod
    def _to_dataframe(records: list[GenerationRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=list(_FRAME_COLS))
        df = pd.DataFrame([r.model_dump() for r in records], columns=list(_FRAME_COLS))
        df["period"] = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        for col in _CATEGORICAL_COLS:
            df[col] = df[col].astype("category")
        return df.sort_values(["period", "fueltype"]).reset_index(drop=True)
