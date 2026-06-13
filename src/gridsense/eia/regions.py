"""EIA balancing-authority / ISO respondent codes.

EIA's v2 API identifies ISOs/RTOs by short respondent codes that don't always
match their common names (e.g. ``CISO`` is the California ISO, commonly "CAISO").
Centralizing them here avoids magic strings across the client, CLI, and future
BigQuery loaders, and lets the CLI validate ``--region`` against a known set.

``StrEnum`` members compare and format exactly as their string value, so a
``Region`` can be passed anywhere a respondent-code string is expected.
"""

from __future__ import annotations

from enum import StrEnum


class Region(StrEnum):
    """Major US ISO/RTO respondent codes in the EIA v2 API."""

    CISO = "CISO"  # California ISO (CAISO)
    ERCO = "ERCO"  # ERCOT — Electric Reliability Council of Texas
    PJM = "PJM"  # PJM Interconnection
    NYIS = "NYIS"  # New York ISO (NYISO)
    ISNE = "ISNE"  # ISO New England
    MISO = "MISO"  # Midcontinent ISO
    SWPP = "SWPP"  # Southwest Power Pool (SPP)
