import logging
import os
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz import fuzz, process
import yfinance as yf

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import MODELS_DIR
from train import predict_next_day, retrain_stock_model
from daily_retrain import run_daily_retrain

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Prediction Engine",
    description="LSTM-based stock price prediction system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STOCKS_FILE = Path(__file__).parent / "stocks" / "master_stocks.json"
os.makedirs(STOCKS_FILE.parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Scheduled daily retrain at 3:00 AM IST
# ---------------------------------------------------------------------------
_last_retrain_result = {"status": "never_run", "last_run": None}


def _run_retrain_job():
    """
    APScheduler job: runs daily retrain and stores the summary.
    Runs in a background thread so API requests are not blocked.
    """
    global _last_retrain_result
    logger.info("Scheduled daily retrain job started.")
    result = run_daily_retrain()
    result["status"] = "completed"
    result["last_run"] = result["timestamp"]
    _last_retrain_result = result


scheduler = BackgroundScheduler()
ist_tz = "Asia/Kolkata"
scheduler.add_job(
    _run_retrain_job,
    trigger=CronTrigger(hour=3, minute=0, timezone=ist_tz),
    id="daily_retrain_3am_ist",
    replace_existing=True,
    misfire_grace_time=3600,  # run within 1 hour if server was down
)


@app.on_event("startup")
def start_scheduler():
    scheduler.start()
    logger.info("APScheduler started — daily retrain scheduled for 3:00 AM IST.")


@app.on_event("shutdown")
def stop_scheduler():
    scheduler.shutdown(wait=False)
    logger.info("APScheduler shut down.")


def load_master_stocks():
    if not STOCKS_FILE.exists():
        return []
    with open(STOCKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_master_stocks(stocks):
    with open(STOCKS_FILE, "w", encoding="utf-8") as f:
        json.dump(stocks, f, indent=2, ensure_ascii=False)


def update_master_stock(new_stock: dict):
    stocks = load_master_stocks()
    exists = any(s["symbol"] == new_stock["symbol"] for s in stocks)
    if not exists:
        stocks.append(new_stock)
        save_master_stocks(stocks)


def fetch_stock_info(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "symbol": symbol.upper(),
        "name": info.get("shortName") or symbol.upper(),
        "exchange": info.get("exchange", "N/A"),
        "country": info.get("country", "N/A"),
        "currency": info.get("currency", "N/A")
    }


class UpdateRequest(BaseModel):
    symbols: List[str]


@app.get("/search")
def search_stocks(q: str = Query(..., min_length=1, description="Company name or partial")):
    """
    Search company name and return top 5 matches.
    Uses prefix and fuzzy matching.
    """
    query = q.lower().strip()
    stocks_list = load_master_stocks()
    results = []

    # Exact or prefix matches
    for stock in stocks_list:
        name_lower = stock["name"].lower()
        stock_symbol = stock["symbol"].lower()
        if name_lower.startswith(query) or stock_symbol.startswith(query):
            results.append(stock)

    # If not enough results, use fuzzy matching
    if len(results) < 5:
        # Build a map: name -> stock
        name_map = {stock["name"]: stock for stock in stocks_list}
        fuzzy_matches = process.extract(query, name_map.keys(), scorer=fuzz.partial_ratio, limit=5)
        for match_name, score, _ in fuzzy_matches:
            stock = name_map[match_name]
            if stock not in results:
                results.append(stock)

    # Return top 5
    return results[:5]


@app.get("/predict")
def predict(symbol: str):
    symbol = symbol.upper().strip()
    try:
        result = predict_next_day(symbol)
        stocks_list = load_master_stocks()
        if not any(s["symbol"] == symbol for s in stocks_list):
            try:
                stock_info = fetch_stock_info(symbol)
            except Exception:
                stock_info = {
                    "symbol": symbol,
                    "name": symbol,
                    "exchange": "N/A",
                    "country": "N/A",
                    "currency": "N/A"
                }
            update_master_stock(stock_info)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/update")
def update_stocks(req: UpdateRequest):
    updated = []

    for symbol in req.symbols:
        try:
            retrain_stock_model(symbol)
            updated.append(symbol.upper())
        except Exception:
            continue

    return {
        "updated_stocks": updated,
        "count": len(updated)
    }


@app.get("/available_stocks")
def available_stocks():
    """
    Return list of stocks that already have trained ONNX models on the server.
    Ensemble members (suffix _ens0/_ens1/...) and scaler JSON sidecars are
    excluded so the extension only shows the primary model per symbol.
    """
    stocks = []
    for file in os.listdir(MODELS_DIR):
        if not file.endswith(".onnx"):
            continue
        symbol = file[:-5].upper()  # strip ".onnx"
        # Skip ensemble members and scaler sidecars
        if "_ENS" in symbol or symbol.endswith("_SCALER"):
            continue
        stocks.append(symbol)
    return {"stocks": stocks}


@app.get("/retrain-status")
def retrain_status():
    """
    Return the status of the last scheduled daily retrain run.
    """
    return _last_retrain_result


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
