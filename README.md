# GridSense

> An agent that turns 30 minutes of energy analyst grunt work into a 30-second question.

![status](https://img.shields.io/badge/status-in%20progress-yellow) ![sprint](https://img.shields.io/badge/built%20during-30%20day%20sprint-blue)

## Problem

Energy analysts at utilities, ISOs, and trading desks spend hours every week answering the same kinds of questions: *"What was CAISO's generation mix yesterday?" "How did wholesale prices move during the heat wave?" "What did FERC say about transmission planning?"* The data is public — EIA, ISO operators, FERC eLibrary — but pulling it requires hopping across portals, downloading CSVs, cross-referencing PDFs, and writing one-off SQL.

GridSense is a multi-agent system that does this in seconds, with citations.

## What it does

Ask a natural-language question. GridSense routes it to the right sub-agent:

- **Data agent** — queries EIA + ISO data in BigQuery via SQL tools
- **Docs agent** — RAG over FERC filings and ISO market notices with grounded citations
- **Forecast agent** — calls a simple forecasting tool for trend questions

Answers come back with the data, a chart where useful, and links to the source filings.

## Architecture (planned)

```
User question
 ↓
Intake / Router agent (Gemini 2.0 or Claude)
 ↓
 ┌──────────────┬──────────────┬──────────────┐
 ↓              ↓              ↓              ↓
Data agent    Docs agent    Forecast agent  Synthesis
(BigQuery)    (RAG/FERC)    (Vertex AI)     (final answer)
```

Deployed on Cloud Run, infrastructure as Terraform, observability via Cloud Trace + structured logging, evaluated against a golden set of analyst questions.

## Quickstart

```bash
# Requires uv (https://docs.astral.sh/uv/) and a free EIA API key:
# https://www.eia.gov/opendata/register.php
uv sync
cp .env.example .env          # paste your EIA_API_KEY

# Fetch a week of hourly generation-by-fuel for the California ISO.
# Note: EIA uses short respondent codes — CAISO is "CISO", ERCOT is "ERCO", etc.
uv run gridsense fetch generation --region CISO --start 2025-06-01 --end 2025-06-07
```

Prints a fuel-mix summary and writes `data/raw/generation_CISO_2025-06-01_2025-06-07.parquet`.
Supported regions: `CISO`, `ERCO`, `PJM`, `NYIS`, `ISNE`, `MISO`, `SWPP`.

## Built during the 30-day sprint

This repo is being built during a 30-day full-time sprint (Data Analyst → Applied AI / Forward Deployed Engineer). Components ship across these days:

| Day | Component | Status |
|---|---|---|
| 2 | EIA ingestion CLI | ☑ |
| 3 | LLM + single tool script (EIA generation tool) | ☐ |
| 4 | Hand-rolled ReAct agent + ADK port, 2 tools | ☐ |
| 5 | RAG bot over FERC filings, with citations | ☐ |
| 8 | Multi-agent orchestration (router + sub-agents) | ☐ |
| 9 | Dockerize + Cloud Run deploy | ☐ |
| 10 | SQL tool over BigQuery (EIA + ISO data) | ☐ |
| 11–13 | Flagship integration: unified multi-agent system | ☐ |
| 15 | Terraform IaC for the full stack | ☐ |
| 16 | Observability: structured logs + Cloud Trace | ☐ |
| 17 | Evaluation pipeline + golden test set | ☐ |
| 19 | Polish + 5-min demo video | ☐ |

## Stack

- **Models:** Gemini 2.0 / Claude 4.x via Vertex AI + Anthropic API
- **Agent framework:** Google ADK (primary), with LangGraph mirror for vocabulary
- **Data:** BigQuery (EIA series + ISO data), Cloud Storage for raw FERC PDFs
- **Vector store:** ChromaDB (local) → Vertex AI Vector Search (production)
- **Deploy:** Docker → Cloud Run
- **IaC:** Terraform
- **Observability:** Cloud Trace + structured logging with conversation ID propagation
- **Evals:** golden set + LLM-as-judge + groundedness scoring

## Data sources

- [EIA Open Data API](https://www.eia.gov/opendata/)
- [CAISO OASIS](http://oasis.caiso.com/)
- [FERC eLibrary](https://elibrary.ferc.gov/)

## Status

🚧 Day 2 of 30 — EIA generation ingestion CLI live (verified against CISO). Components ship as the calendar advances.

## License

MIT
