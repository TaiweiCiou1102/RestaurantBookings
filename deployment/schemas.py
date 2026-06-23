"""Pydantic schemas for the three-subset AML scoring endpoint.

Request format mirrors example_input.json:
    {
      "input_data": {
        "columns": ["reservation_hour", "party_size", "dining_purpose", ...],
        "index":   [0, 1, 2],
        "data":    [[19, 3, "家人用餐", "F", ...], ...]
      }
    }

`reservation_hour` is required: score.py uses it to route each row to the
lunch / afternoon / dinner sub-model, then drops it before prediction.

Usage:
    req = ScoringRequest.model_validate_json(raw_json)
    df  = req.to_dataframe()          # ready for score.run()
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, model_validator


# ── Single booking row (typed, for documentation / validation) ─────────────

class BookingRecord(BaseModel):
    """One row of input features — types reflect what the model was trained on.

    Matches the 25 model features plus ``reservation_hour`` (routing only).
    Geographic codes (city_code / city_area_code / member_city_code) were dropped
    from the model on 2026-06-22 — see docs/feature_decisions.md.
    """

    # Routing (which meal period) — not a model feature, dropped before predict
    reservation_hour:     int

    # Booking details
    party_size:           int
    dining_purpose:       str
    booked_by_gender:     str
    prepay_satisfied:     int

    # Restaurant attributes
    hotel_restaurant:     float
    family_friendly:      float
    accepts_credit_card:  float
    has_parking:          float
    outdoor_seating:      float
    has_wifi:             float
    wheelchair_accessible: float
    lat:                  float
    lng:                  float

    # Member attributes
    is_vip_member:        float
    account_gender:       str

    # Derived / temporal features
    lead_time:            float
    age:                  float
    booking_hour:         float
    weekday:              int
    avg_price:            float
    is_holiday_vicinity:  int
    days_from_payday:     float
    popular_hour:         int
    dinner_ratio:         float
    restaurant_density:   int


# ── AML payload format ─────────────────────────────────────────────────────

class InputData(BaseModel):
    """The `input_data` block in AML scoring format."""

    columns: list[str]
    index:   list[int]
    data:    list[list[Any]]

    @model_validator(mode="after")
    def _check_dimensions(self) -> InputData:
        n_cols = len(self.columns)
        if len(self.index) != len(self.data):
            raise ValueError(
                f"index length ({len(self.index)}) != data length ({len(self.data)})"
            )
        for i, row in enumerate(self.data):
            if len(row) != n_cols:
                raise ValueError(
                    f"row {i} has {len(row)} values but columns declares {n_cols}"
                )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=self.columns, index=self.index)


class ScoringRequest(BaseModel):
    """Top-level request body sent to the AML endpoint."""

    input_data: InputData

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame ready to pass into score.run()."""
        return self.input_data.to_dataframe()


# ── Response format ────────────────────────────────────────────────────────

class PredictionData(BaseModel):
    # columns: ["reservation_half_hour", "time_range", "subset", "probability"]
    columns: list[str]
    index:   list[int]
    # one row per input: [slot, "HH:MM-HH:MM", subset, probability]
    # null for rows outside every meal period (subset == "_unmatched")
    data:    list[list[Optional[Any]]]


class ScoringResponse(BaseModel):
    """Response body returned by the AML endpoint."""

    predictions: PredictionData
