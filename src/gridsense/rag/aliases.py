"""Short, human-friendly citation labels for source PDFs.

Citations render as ``[<alias> p.<n>]``. The filename (not the alias) is what we
store in Chroma metadata, so changing a label here never requires re-ingesting.
"""

from __future__ import annotations

from pathlib import Path

ALIAS_MAP = {
    "20260604-5190_Petition for Waiver - Silver Star.pdf": "Silver Star",
    "20260615-5151_Joint Index of Exhibits DCR Transmission ER23-2309 et al (2026-06-15).pdf": (
        "ER23-2309 Index"
    ),
    "may2026.pdf": "EIA EPM May 2026",
}


def alias_for(source_pdf: str) -> str:
    """Citation label for a source filename; falls back to the filename stem."""
    return ALIAS_MAP.get(source_pdf, Path(source_pdf).stem)
