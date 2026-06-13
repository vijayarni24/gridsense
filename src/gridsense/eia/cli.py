"""Typer CLI: ``gridsense fetch generation --region CAISO --start ... --end ...``.

Typer is synchronous; the async client is bridged with a single ``asyncio.run``
call at this boundary so async never leaks into the UI layer.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from dotenv import load_dotenv

from gridsense.eia.client import EIAClient

app = typer.Typer(add_completion=False, help="GridSense data CLI")
fetch_app = typer.Typer(help="Fetch data from upstream sources")
app.add_typer(fetch_app, name="fetch")


async def _fetch(api_key: str, region: str, start: datetime, end: datetime) -> pd.DataFrame:
    async with EIAClient(api_key) as client:
        return await client.fetch_generation(region, start.date(), end.date())


def _print_summary(df: pd.DataFrame, region: str, out_path: Path) -> None:
    typer.secho(f"\n✓ {region}: {len(df):,} rows", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  period:  {df['period'].min()} → {df['period'].max()}")
    typer.echo(f"  written: {out_path}")

    by_fuel = (
        df.groupby("type_name", observed=True)["value"].sum().sort_values(ascending=False)
    )
    total = by_fuel.sum()
    typer.secho("\n  generation by fuel (sum over range):", bold=True)
    for fuel, mwh in by_fuel.items():
        share = (mwh / total * 100) if total else 0.0
        typer.echo(f"    {str(fuel):<22} {mwh:>14,.0f} MWh  ({share:4.1f}%)")


@fetch_app.command("generation")
def fetch_generation(
    region: Annotated[str, typer.Option("--region", help="ISO/RTO respondent code, e.g. CAISO")],
    start: Annotated[datetime, typer.Option("--start", formats=["%Y-%m-%d"], help="YYYY-MM-DD")],
    end: Annotated[
        datetime,
        typer.Option("--end", formats=["%Y-%m-%d"], help="YYYY-MM-DD (inclusive)"),
    ],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Parquet output directory")
    ] = Path("data/raw"),
) -> None:
    """Fetch hourly net generation by fuel type and write parquet to data/raw/."""
    load_dotenv()
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        typer.secho(
            "EIA_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free: https://www.eia.gov/opendata/register.php).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if end < start:
        typer.secho("--end must not be before --start", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = asyncio.run(_fetch(api_key, region, start, end))
    if df.empty:
        typer.secho(f"No data returned for {region} in that range.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generation_{region}_{start.date()}_{end.date()}.parquet"
    df.to_parquet(out_path, index=False)

    _print_summary(df, region, out_path)


if __name__ == "__main__":
    app()
