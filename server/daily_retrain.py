"""
Daily retrain job — updates all trained stock models at 3:00 AM IST,
after all major global markets have closed and data has propagated.

This does NOT wipe existing data or models (unlike rebuild_models.py).
It fetches the latest data, appends to CSVs, retrains, and re-exports ONNX.
"""

import json
import logging
import os
from datetime import datetime, timezone

from train import train_new_stock
from config import MODELS_DIR

logger = logging.getLogger(__name__)


def get_trained_symbols() -> list[str]:
    """Return a list of symbols that already have ONNX models on disk.

    Ensemble members (suffix _ens0/_ens1/...) and scaler JSON sidecars are
    excluded so the daily retrain only touches the primary model per symbol.
    """
    symbols = []
    for file in os.listdir(MODELS_DIR):
        if not file.endswith(".onnx"):
            continue
        symbol = file[:-5].upper()  # strip ".onnx"
        if "_ENS" in symbol or symbol.endswith("_SCALER"):
            continue
        symbols.append(symbol)
    return symbols


def load_master_stock_symbols() -> list[str]:
    """Load symbols from master_stocks.json (fallback if no models exist yet)."""
    stocks_file = os.path.join(os.path.dirname(__file__), "stocks", "master_stocks.json")
    if not os.path.exists(stocks_file):
        return []
    with open(stocks_file, "r", encoding="utf-8") as f:
        stocks = json.load(f)
    return [s["symbol"].upper() for s in stocks]


def run_daily_retrain() -> dict:
    """
    Retrain all models that have existing ONNX artifacts.
    Returns a summary dict with counts and per-symbol results.
    """
    results = []
    success_count = 0
    fail_count = 0

    symbols = get_trained_symbols()
    if not symbols:
        logger.info("No ONNX models found on disk — trying master_stocks.json")
        symbols = load_master_stock_symbols()

    if not symbols:
        logger.warning("No symbols to retrain. Master stock list is also empty.")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": 0,
            "success": 0,
            "fail": 0,
            "results": [],
        }

    logger.info("Starting daily retrain for %d symbol(s): %s", len(symbols), ", ".join(symbols))

    for sym in symbols:
        try:
            logger.info("Retraining %s ...", sym)
            # train_new_stock uses EPOCHS_NEW (100) by default for best quality
            train_new_stock(sym)
            results.append({"symbol": sym, "status": "ok"})
            success_count += 1
            logger.info("Retrain of %s succeeded", sym)
        except Exception as e:
            logger.exception("Retrain of %s failed", sym)
            results.append({"symbol": sym, "status": "error", "error": str(e)})
            fail_count += 1

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(symbols),
        "success": success_count,
        "fail": fail_count,
        "results": results,
    }

    logger.info(
        "Daily retrain complete. %d succeeded, %d failed out of %d total.",
        success_count, fail_count, len(symbols),
    )

    return summary
