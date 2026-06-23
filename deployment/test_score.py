"""Local test for the three-subset score.py — no Azure required.

Points AZUREML_MODEL_DIR at deployment/ (which contains model_bundle/), then
exercises init() + run() exactly as the Azure ML inference server would.

Run from anywhere (paths are resolved relative to this file):
    uv run python deployment/test_score.py

Prereq: run `uv run python deployment/export_bundle.py` first to build the bundle.

Tests:
    1. init()          — all three sub-models load
    2. run()           — example_input.json returns valid predictions JSON
    3. routing         — each row is scored by the expected meal-period sub-model
    4. output values   — slot is within the routed subset, prob in [0, 1]
    5. unmatched hour  — reservation_hour outside all periods -> null / "_unmatched"
    6. missing route   — input without reservation_hour raises a clear error
    7. unseen category — unknown category value does not crash
"""

import json
import os
import sys
from pathlib import Path

# ── Make `import score` work and point AZUREML_MODEL_DIR at the bundle ───────
DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_DIR))
os.environ["AZUREML_MODEL_DIR"] = str(DEPLOY_DIR)  # contains model_bundle/

import score  # noqa: E402

EXAMPLE_INPUT = DEPLOY_DIR / "example_input.json"

# Expected hour -> subset routing (from configs/xgboost.yaml)
EXPECTED_ROUTE = {12: "lunch", 16: "afternoon", 19: "dinner"}


def ok(msg):   print(f"  PASS  {msg}")
def fail(msg): print(f"  FAIL  {msg}"); sys.exit(1)


def test_init():
    try:
        score.init()
        assert score._ROUTING is not None, "_ROUTING is None"
        assert len(score._SUBSETS) == 3, f"expected 3 sub-models, got {len(score._SUBSETS)}"
        ok(f"init() — loaded sub-models: {sorted(score._SUBSETS)}")
    except Exception as e:
        fail(f"init() raised {type(e).__name__}: {e}")


def test_run_example():
    raw = EXAMPLE_INPUT.read_text(encoding="utf-8")
    try:
        out = json.loads(score.run(raw))
    except Exception as e:
        fail(f"run() raised {type(e).__name__}: {e}")

    p = out.get("predictions", {})
    for k in ("columns", "index", "data"):
        assert k in p, f"missing 'predictions.{k}'"
    assert p["columns"] == ["reservation_half_hour", "time_range", "subset", "probability"], \
        f"unexpected columns: {p['columns']}"
    ok("run() — output JSON structure is valid")
    return out


def test_routing_and_values(out):
    raw = json.loads(EXAMPLE_INPUT.read_text(encoding="utf-8"))
    hours = {idx: row[0] for idx, row in zip(raw["input_data"]["index"], raw["input_data"]["data"])}
    cols = out["predictions"]["columns"]
    si, sloti, probi = cols.index("subset"), cols.index("reservation_half_hour"), cols.index("probability")

    for idx, row in zip(out["predictions"]["index"], out["predictions"]["data"]):
        hr = hours[idx]
        exp_subset = EXPECTED_ROUTE[hr]
        assert row[si] == exp_subset, f"row {idx}: hour {hr} routed to {row[si]}, expected {exp_subset}"
        meta = score._ROUTING["subsets"][exp_subset]
        slot_hour = int(row[sloti]) // 2
        assert meta["hour_min"] <= slot_hour <= meta["hour_max"], \
            f"row {idx}: predicted slot {row[sloti]} (hour {slot_hour}) outside {exp_subset}"
        assert 0.0 <= row[probi] <= 1.0, f"row {idx}: probability {row[probi]} out of range"
    ok(f"routing — all rows scored by correct sub-model {EXPECTED_ROUTE}")


def _base_columns():
    return score._ROUTING["feature_columns"]


def _payload(hour, overrides=None):
    """Build a one-row payload (reservation_hour + all feature columns)."""
    defaults = {
        "party_size": 2, "dining_purpose": "生日慶祝", "booked_by_gender": "M",
        "prepay_satisfied": 1, "hotel_restaurant": 0.0, "family_friendly": 1.0,
        "accepts_credit_card": 1.0, "has_parking": 1.0, "outdoor_seating": 1.0,
        "has_wifi": 0.0, "wheelchair_accessible": 0.0, "lat": 25.03, "lng": 121.54,
        "is_vip_member": 0.0, "account_gender": "M", "lead_time": 48.0, "age": 28.0,
        "booking_hour": 18.0, "weekday": 5, "avg_price": 750.0, "is_holiday_vicinity": 0,
        "days_from_payday": 10.0, "popular_hour": 19, "dinner_ratio": 0.72,
        "restaurant_density": 30,
    }
    if overrides:
        defaults.update(overrides)
    cols = ["reservation_hour"] + _base_columns()
    vals = [hour] + [defaults[c] for c in _base_columns()]
    return json.dumps({"input_data": {"columns": cols, "index": [0], "data": [vals]}}, ensure_ascii=False)


def test_unmatched_hour():
    # hour 5 falls outside all meal periods (11-23)
    out = json.loads(score.run(_payload(5)))
    row = out["predictions"]["data"][0]
    si = out["predictions"]["columns"].index("subset")
    assert row[si] == "_unmatched", f"expected _unmatched, got {row[si]}"
    assert row[0] is None, "unmatched row should have null slot"
    ok("unmatched hour — null prediction with subset '_unmatched'")


def test_missing_route_column():
    cols = _base_columns()  # deliberately omit reservation_hour
    vals = [2, "生日慶祝", "M", 1, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 25.03, 121.54,
            0.0, "M", 48.0, 28.0, 18.0, 5, 750.0, 0, 10.0, 19, 0.72, 30]
    payload = json.dumps({"input_data": {"columns": cols, "index": [0], "data": [vals]}}, ensure_ascii=False)
    try:
        score.run(payload)
        fail("missing reservation_hour should raise, but did not")
    except ValueError:
        ok("missing route column — raises ValueError as expected")


def test_unseen_category():
    out = json.loads(score.run(_payload(19, {"dining_purpose": "UNSEEN_PURPOSE"})))
    row = out["predictions"]["data"][0]
    assert row[0] is not None, "unseen category should still predict"
    ok(f"unseen category — handled, predicted {row[1]} (subset {row[2]})")


if __name__ == "__main__":
    print("=" * 55)
    print("  score.py (three-subset) — local test")
    print("=" * 55)

    test_init()
    out = test_run_example()
    test_routing_and_values(out)
    test_unmatched_hour()
    test_missing_route_column()
    test_unseen_category()

    print()
    print("  All tests passed.")
    print("=" * 55)

    # ── I/O example ──────────────────────────────────────────────────────────
    print("\n  I/O Example  (example_input.json)\n" + "=" * 55)
    pred = out["predictions"]
    for idx, row in zip(pred["index"], pred["data"]):
        print(f"  row {idx}:  slot={row[0]}  time={row[1]}  subset={row[2]}  prob={row[3]}")
    print("=" * 55)
