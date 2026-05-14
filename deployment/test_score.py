"""Local test for score.py — uses artifact paths from work.ipynb.

Run from the project root:
    uv run python deployment/test_score.py

Tests:
    1. init()        — model and encoders load without error
    2. run()         — example_input.json returns valid predictions JSON
    3. output shape  — shape (n_rows, n_classes): one probability row per input row
    4. probability dist — columns are actual class labels; each row sums to ~1.0
    5. unseen category — gracefully maps unknown category to -1, no crash
    6. single row    — single-row input works (no squeeze/index bug)
"""

import json
import os
import sys

# ── Run from project root regardless of how the script is invoked ──────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import score  # imports deployment/score.py via sys.path — run from this dir

# ── Artifact paths (matching work.ipynb cells 10) ──────────────────────────
MODEL_PATH      = "mlruns/2/models/m-858ece605d8449a7a921a8385df78186/artifacts/model.ubj"
X_ENCODER_PATH  = "mlruns/2/0c88e1327c5441cc8b8ac574045e3fce/artifacts/encoders.joblib"
Y_ENCODER_PATH  = "mlruns/2/0c88e1327c5441cc8b8ac574045e3fce/artifacts/target_encoder.joblib"

EXAMPLE_INPUT   = os.path.join(os.path.dirname(__file__), "example_input.json")


# ── Helpers ────────────────────────────────────────────────────────────────

def _patch_init_and_run():
    """Monkey-patch init() to use absolute paths instead of AZUREML_MODEL_DIR."""
    import joblib
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    score._MODEL     = model
    score._X_ENCODER = joblib.load(X_ENCODER_PATH)
    score._Y_ENCODER = joblib.load(Y_ENCODER_PATH)


def ok(msg):  print(f"  PASS  {msg}")
def fail(msg): print(f"  FAIL  {msg}"); sys.exit(1)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_init():
    """Artifacts load without raising exceptions."""
    try:
        _patch_init_and_run()
        assert score._MODEL     is not None, "_MODEL is None"
        assert score._X_ENCODER is not None, "_X_ENCODER is None"
        assert score._Y_ENCODER is not None, "_Y_ENCODER is None"
        ok("init() — model and encoders loaded")
    except Exception as e:
        fail(f"init() raised {type(e).__name__}: {e}")


def test_run_example_input():
    """run() on example_input.json returns valid JSON with correct structure."""
    with open(EXAMPLE_INPUT, encoding="utf-8") as f:
        raw = f.read()

    try:
        result = score.run(raw)
    except Exception as e:
        fail(f"run() raised {type(e).__name__}: {e}")

    try:
        out = json.loads(result)
    except json.JSONDecodeError as e:
        fail(f"Output is not valid JSON: {e}")

    assert "predictions" in out,          "missing 'predictions' key"
    assert "columns" in out["predictions"], "missing 'predictions.columns'"
    assert "data"    in out["predictions"], "missing 'predictions.data'"
    assert "index"   in out["predictions"], "missing 'predictions.index'"
    ok("run() — output JSON structure is valid")
    return out


def test_output_shape(out):
    """One probability row per input row; width equals number of model classes."""
    with open(EXAMPLE_INPUT, encoding="utf-8") as f:
        n_input = len(json.load(f)["input_data"]["data"])

    n_pred    = len(out["predictions"]["data"])
    n_classes = len(out["predictions"]["columns"])
    assert n_pred == n_input, f"expected {n_input} rows, got {n_pred}"
    for i, row in enumerate(out["predictions"]["data"]):
        assert len(row) == n_classes, f"row {i}: expected {n_classes} values, got {len(row)}"
    ok(f"output shape — ({n_pred}, {n_classes})  [rows × classes]")


def test_prediction_range(out):
    """Columns are time-range strings matching the model's classes; each row sums to ~1.0."""
    known = [score._convert_class_to_time_range(int(c)) for c in score._Y_ENCODER.classes_]
    assert set(out["predictions"]["columns"]) == set(known), (
        f"columns {out['predictions']['columns']} != expected {sorted(known)}"
    )
    for i, row in enumerate(out["predictions"]["data"]):
        assert len(row) == len(known), f"row {i}: expected {len(known)} probs, got {len(row)}"
        total = sum(row)
        assert abs(total - 1.0) < 1e-4, f"row {i}: probabilities sum to {total:.6f}, expected ~1.0"
        assert all(0.0 <= p <= 1.0 for p in row), f"row {i}: probability out of [0, 1] range"
    ok(f"prediction range — {len(known)} classes, all rows sum to 1.0")


def test_unseen_category():
    """Unseen category value does not crash; returns a valid hour."""
    payload = {
        "input_data": {
            "columns": [
                "party_size", "dining_purpose", "booked_by_gender", "prepay_satisfied",
                "hotel_restaurant", "city_code", "city_area_code", "family_friendly",
                "accepts_credit_card", "has_parking", "outdoor_seating", "has_wifi",
                "wheelchair_accessible", "lat", "lng", "is_vip_member", "account_gender",
                "member_city_code", "lead_time", "age", "booking_hour", "weekday",
                "avg_price", "is_holiday_vicinity", "days_from_payday", "popular_hour",
                "dinner_ratio", "restaurant_density",
            ],
            "index": [0],
            "data": [[
                4, "UNSEEN_PURPOSE", "F", 1, 1.0, "UNSEEN_CITY", "UNSEEN_AREA",
                1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 25.05, 121.52, 0.0, "F", "Unknown",
                120.0, 30.0, 12.0, 3, 600.0, 0, 5.0, 12, 0.5, 20,
            ]],
        }
    }
    try:
        result = score.run(json.dumps(payload, ensure_ascii=False))
        out = json.loads(result)
        row = out["predictions"]["data"][0]
        assert abs(sum(row) - 1.0) < 1e-4, f"probabilities sum to {sum(row):.6f}"
        ok(f"unseen category — handled gracefully, top-1 hour = {out['predictions']['columns'][row.index(max(row))]}")
    except Exception as e:
        fail(f"unseen category raised {type(e).__name__}: {e}")


def test_single_row():
    """Single-row input returns exactly one prediction (no index/squeeze bug)."""
    payload = {
        "input_data": {
            "columns": [
                "party_size", "dining_purpose", "booked_by_gender", "prepay_satisfied",
                "hotel_restaurant", "city_code", "city_area_code", "family_friendly",
                "accepts_credit_card", "has_parking", "outdoor_seating", "has_wifi",
                "wheelchair_accessible", "lat", "lng", "is_vip_member", "account_gender",
                "member_city_code", "lead_time", "age", "booking_hour", "weekday",
                "avg_price", "is_holiday_vicinity", "days_from_payday", "popular_hour",
                "dinner_ratio", "restaurant_density",
            ],
            "index": [0],
            "data": [[
                2, "生日慶祝", "M", 1, 0.0, "台北市", "大安區",
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 25.03, 121.54, 0.0, "M", "台北市",
                48.0, 28.0, 18.0, 5, 750.0, 0, 10.0, 19, 0.72, 30,
            ]],
        }
    }
    result = score.run(json.dumps(payload, ensure_ascii=False))
    out = json.loads(result)
    assert len(out["predictions"]["data"]) == 1, "expected exactly 1 prediction row"
    row = out["predictions"]["data"][0]
    top_hour = out["predictions"]["columns"][row.index(max(row))]
    ok(f"single row — top-1 hour = {top_hour} (prob = {max(row):.4f})")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # score.py must be importable → run from deployment/
    # but artifact paths are relative to project root → chdir there first,
    # then add deployment/ to sys.path so `import score` still works.
    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_ROOT)
    if deployment_dir not in sys.path:
        sys.path.insert(0, deployment_dir)

    print("=" * 55)
    print("  score.py — local test")
    print("=" * 55)

    test_init()
    out = test_run_example_input()
    test_output_shape(out)
    test_prediction_range(out)
    test_unseen_category()
    test_single_row()

    print()
    print("  All tests passed.")
    print("=" * 55)

    # ── I/O example ────────────────────────────────────────────────────────
    print()
    print("  I/O Example")
    print("=" * 55)

    with open(EXAMPLE_INPUT, encoding="utf-8") as f:
        example = json.load(f)

    cols    = example["input_data"]["columns"]
    indices = example["input_data"]["index"]
    rows    = example["input_data"]["data"]

    print("  [INPUT]")
    col_w = max(len(c) for c in cols)
    for idx, row in zip(indices, rows):
        print(f"  row {idx}:")
        for c, v in zip(cols, row):
            print(f"    {c:<{col_w}} = {v}")

    print()
    print("  [OUTPUT]  (time slot → probability)")
    pred = out["predictions"]
    for idx, row in zip(pred["index"], pred["data"]):
        top_i    = row.index(max(row))
        top_time = pred["columns"][top_i]
        top_prob = row[top_i]
        print(f"  row {idx}:  top-1 = {top_time}  prob={top_prob:.4f}")
        for time_label, prob in zip(pred["columns"], row):
            bar = "#" * int(prob * 40)
            print(f"    {time_label}  {prob:.4f}  {bar}")
    print("=" * 55)
