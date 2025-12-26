from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from config import MODELS_DIR
import os
import json
from fastapi import Query
from rapidfuzz import fuzz, process
from pathlib import Path
import yfinance as yf

from train import predict_next_day, update_stock_model

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
        "name": info.get("shortName", symbol.upper()) or symbol.upper(),
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
            update_stock_model(symbol)
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
    Return list of stocks that already have trained models on the server.
    """
    stocks = []
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pt"):
            symbol = file.replace(".pt", "")
            stocks.append(symbol)
    return {"stocks": stocks}


@app.get("/health")
def health():
    return {"status": "ok"}


