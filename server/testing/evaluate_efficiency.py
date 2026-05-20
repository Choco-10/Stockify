import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, MODELS_DIR, SEQ_LENGTH
from train import inverse_with_metadata, load_model, normalize_with_metadata
from utils import fetch_stock_data


def discover_symbols():
    symbols = []
    for name in os.listdir(MODELS_DIR):
        if name.endswith(".onnx"):
            symbols.append(name[:-5].upper())
    return sorted(symbols)


def load_prices(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None

    # Try to read a headered CSV first and detect a Close column.
    try:
        df = pd.read_csv(path)
        # Filter out the most recent rows (keep only up to today - 2 days)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        # detect a Date column (case-insensitive)
        date_col = None
        for c in df.columns:
            if str(c).strip().lower() == "date":
                date_col = c
                break
        if date_col is not None:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df[df[date_col].dt.date <= cutoff_date]
            except Exception:
                pass
        # Normalize common Close column names
        cols = [c for c in df.columns]
        close_col = None
        for c in cols:
            if str(c).strip().lower() == "close":
                close_col = c
                break
        if close_col is not None and df[close_col].size > 0:
            prices = pd.to_numeric(df[close_col], errors="coerce").dropna().values.astype(np.float32)
            if prices.size > 0:
                return prices.reshape(-1, 1)
    except Exception:
        # Fall through to legacy parsing below
        pass

    # Legacy fallback: CSV without header, assume second column contains price
    try:
        df = pd.read_csv(path, header=None)
        # attempt to parse first column as date and trim latest 2 days if possible
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        try:
            parsed = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            if parsed.notna().any():
                mask = parsed.dt.date <= cutoff_date
                df = df[mask]
        except Exception:
            pass

        if df.shape[1] < 2 or df.empty:
            return None
        prices = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().values.astype(np.float32)
        if prices.size == 0:
            return None
        return prices.reshape(-1, 1)
    except Exception:
        return None


def percentile(values, pct):
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def label_mape(mape_percent):
    if mape_percent < 1.5:
        return "very_strong"
    if mape_percent < 3.0:
        return "good"
    if mape_percent <= 5.0:
        return "acceptable"
    if mape_percent <= 8.0:
        return "poor"
    return "worst"


def label_direction(direction_percent):
    if direction_percent >= 60.0:
        return "very_strong"
    if direction_percent >= 55.0:
        return "good"
    if direction_percent >= 50.0:
        return "acceptable"
    if direction_percent >= 45.0:
        return "poor"
    return "worst"


def label_latency(avg_latency_ms):
    if avg_latency_ms < 5.0:
        return "excellent"
    if avg_latency_ms <= 20.0:
        return "good"
    if avg_latency_ms <= 50.0:
        return "acceptable"
    if avg_latency_ms <= 100.0:
        return "poor"
    return "worst"


def direction_binary(returns):
    # Strict up/down labeling: zero is treated as up to keep binary classes only.
    return np.where(returns >= 0.0, 1, -1)


def evaluate_symbol(symbol, test_size=120, warmup=10):
    scaler_meta, onnx_session = load_model(symbol)
    if scaler_meta is None or onnx_session is None:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": "Model or scaler metadata missing",
        }

    # Determine whether the model expects multi-feature input
    model_num_features = int(scaler_meta.get("num_features", 1))

    # For single-feature models we can use legacy CSV parsing (Close-only)
    if model_num_features == 1:
        prices = load_prices(symbol)
        if prices is None:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "No local CSV data found",
            }

        if len(prices) <= SEQ_LENGTH + 1:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "Not enough history for sequence evaluation",
            }

        start_idx = max(SEQ_LENGTH, len(prices) - test_size)

        preds = []
        actuals = []
        prev_actuals = []
        latencies_ms = []

        input_name = onnx_session.get_inputs()[0].name

        for idx in range(start_idx, len(prices)):
            seq_raw = prices[idx - SEQ_LENGTH:idx]
            # Skip sequences with missing values
            if np.isnan(seq_raw).any():
                continue
            seq_norm = normalize_with_metadata(seq_raw, scaler_meta)
            model_input = np.expand_dims(seq_norm.astype(np.float32), axis=0)

            t0 = time.perf_counter()
            pred_out = onnx_session.run(None, {input_name: model_input})[0]
            if np.isnan(pred_out).any():
                continue
            t1 = time.perf_counter()

            actual_price = float(prices[idx][0])
            prev_price = float(prices[idx - 1][0])

            target_type = scaler_meta.get("target_type", "price_norm")
            if target_type == "return":
                pred_return = float(pred_out[0][0])
                pred_price = float(prev_price * (1.0 + pred_return))
            else:
                pred_price = float(inverse_with_metadata(pred_out, scaler_meta)[0][0])

            # Skip if actuals are invalid
            if math.isnan(actual_price) or math.isnan(prev_price):
                continue
            preds.append(pred_price)
            actuals.append(actual_price)
            prev_actuals.append(prev_price)
            latencies_ms.append((t1 - t0) * 1000.0)
    else:
        # Multi-feature model: attempt to load full OHLCV DataFrame
        try:
            df = fetch_stock_data(symbol, save_csv=False)
        except Exception:
            # If fetch fails, try reading local CSV directly
            path = os.path.join(DATA_DIR, f"{symbol}.csv")
            try:
                df = pd.read_csv(path)
            except Exception:
                return {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "Unable to load multi-feature CSV data",
                }

        # Trim to exclude the most recent 2 days (today and yesterday)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        try:
            # If Date is an index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index.date <= cutoff_date]
            else:
                # detect Date column
                date_col = None
                for c in df.columns:
                    if str(c).strip().lower() == "date":
                        date_col = c
                        break
                if date_col is not None:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                        df = df[df[date_col].dt.date <= cutoff_date]
                    except Exception:
                        pass
        except Exception:
            pass

        # Ensure required feature columns exist
        feature_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if len(feature_cols) != model_num_features:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": f"Model expects {model_num_features} features but CSV has {len(feature_cols)}",
            }
        if "Close" not in df.columns or len(df) <= SEQ_LENGTH + 1:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "Not enough history for sequence evaluation",
            }

        features = df[feature_cols].values.astype(np.float32)
        close_vals = df["Close"].astype(float).values.astype(np.float32)

        start_idx = max(SEQ_LENGTH, len(features) - test_size)

        preds = []
        actuals = []
        prev_actuals = []
        latencies_ms = []

        input_name = onnx_session.get_inputs()[0].name

        for idx in range(start_idx, len(features)):
            seq_raw = features[idx - SEQ_LENGTH:idx]
            # Skip sequences with missing values
            if np.isnan(seq_raw).any():
                continue
            seq_norm = normalize_with_metadata(seq_raw, scaler_meta)
            model_input = np.expand_dims(seq_norm.astype(np.float32), axis=0)

            t0 = time.perf_counter()
            pred_out = onnx_session.run(None, {input_name: model_input})[0]
            if np.isnan(pred_out).any():
                continue
            t1 = time.perf_counter()

            actual_price = float(close_vals[idx])
            prev_price = float(close_vals[idx - 1])

            target_type = scaler_meta.get("target_type", "price_norm")
            if target_type == "return":
                pred_return = float(pred_out[0][0])
                pred_price = float(prev_price * (1.0 + pred_return))
            else:
                pred_price = float(inverse_with_metadata(pred_out, scaler_meta)[0][0])

            # Skip if actuals are invalid
            if math.isnan(actual_price) or math.isnan(prev_price):
                continue
            preds.append(pred_price)
            actuals.append(actual_price)
            prev_actuals.append(prev_price)
            latencies_ms.append((t1 - t0) * 1000.0)

    preds_np = np.array(preds, dtype=np.float64)
    actuals_np = np.array(actuals, dtype=np.float64)
    prev_np = np.array(prev_actuals, dtype=np.float64)

    # If no valid prediction samples were collected, skip this symbol
    if preds_np.size == 0:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": "No valid prediction samples after trimming or due to NaNs",
        }

    abs_err = np.abs(preds_np - actuals_np)
    mae = float(np.mean(abs_err))
    rmse = float(math.sqrt(np.mean((preds_np - actuals_np) ** 2)))

    safe_actual = np.where(np.abs(actuals_np) < 1e-8, np.nan, actuals_np)
    mape = float(np.nanmean(np.abs((preds_np - actuals_np) / safe_actual)) * 100.0)

    denom = np.abs(preds_np) + np.abs(actuals_np)
    smape = float(np.nanmean(np.where(denom < 1e-8, np.nan, (2.0 * np.abs(preds_np - actuals_np) / denom))) * 100.0)

    # Convert moves to returns and force binary direction labels (up/down).
    safe_prev = np.where(np.abs(prev_np) < 1e-8, np.nan, prev_np)
    pred_returns = (preds_np - prev_np) / safe_prev
    actual_returns = (actuals_np - prev_np) / safe_prev

    valid_mask = ~np.isnan(pred_returns) & ~np.isnan(actual_returns)
    pred_direction = direction_binary(pred_returns[valid_mask])
    actual_direction = direction_binary(actual_returns[valid_mask])

    directional_accuracy = float(np.mean(pred_direction == actual_direction) * 100.0)

    timing_slice = latencies_ms[min(warmup, len(latencies_ms)):]
    if not timing_slice:
        timing_slice = latencies_ms

    avg_latency_ms = float(np.mean(timing_slice))
    p95_latency_ms = percentile(timing_slice, 95)
    throughput = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else float("inf")

    return {
        "symbol": symbol,
        "status": "ok",
        "samples": int(len(preds_np)),
        "target_type": scaler_meta.get("target_type", "price_norm"),
        "metrics": {
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
            "MAPE_percent": round(mape, 6),
            "sMAPE_percent": round(smape, 6),
            "directional_accuracy_percent": round(directional_accuracy, 6),
            "avg_latency_ms": round(avg_latency_ms, 6),
            "p95_latency_ms": round(p95_latency_ms, 6),
            "throughput_preds_per_sec": round(throughput, 6),
        },
        "labels": {
            "price_accuracy": label_mape(mape),
            "direction_accuracy": label_direction(directional_accuracy),
            "latency": label_latency(avg_latency_ms),
        },
    }


def summarize(results):
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return {}

    keys = [
        "MAPE_percent",
        "sMAPE_percent",
        "directional_accuracy_percent",
    ]

    summary = {}
    for key in keys:
        vals = [r["metrics"][key] for r in ok_results]
        summary[key] = round(float(np.mean(vals)), 6)

    summary["symbols_tested"] = len(ok_results)
    summary["labels"] = {
        "price_accuracy": label_mape(summary["MAPE_percent"]),
        "direction_accuracy": label_direction(summary["directional_accuracy_percent"]),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate stock model prediction quality and runtime efficiency")
    parser.add_argument("--symbols", nargs="*", help="Symbols to evaluate (default: auto-discover from models/*.onnx)")
    parser.add_argument("--test-size", type=int, default=120, help="Number of latest points used for temporal holdout")
    parser.add_argument("--warmup", type=int, default=10, help="Number of initial inferences excluded from latency stats")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else discover_symbols()
    if not symbols:
        raise RuntimeError("No symbols found. Train at least one model first.")

    results = []
    for symbol in symbols:
        results.append(
            evaluate_symbol(
                symbol,
                test_size=args.test_size,
                warmup=args.warmup,
            )
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "SEQ_LENGTH": SEQ_LENGTH,
            "test_size": args.test_size,
            "warmup": args.warmup,
        },
        "summary": summarize(results),
        "symbols": results,
    }

    reports_dir = os.path.join("testing", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(reports_dir, f"efficiency_report_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
