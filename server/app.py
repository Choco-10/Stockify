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

# Load master stock list once at startup
STOCKS_FILE = Path(__file__).parent / "stocks" / "master_stocks.json"
with open(STOCKS_FILE, "r", encoding="utf-8") as f:
    MASTER_STOCK_LIST = json.load(f)


class UpdateRequest(BaseModel):
    symbols: List[str]


@app.get("/search")
def search_stocks(q: str = Query(..., min_length=1, description="Company name or partial")):
    """
    Search company name and return top 5 matches.
    Uses prefix and fuzzy matching.
    """
    query = q.lower().strip()
    results = []

    # Exact or prefix matches
    for stock in MASTER_STOCK_LIST:
        name_lower = stock["name"].lower()
        if query in name_lower:
            results.append(stock)

    # If not enough results, use fuzzy matching
    if len(results) < 5:
        # Build a map: name -> stock
        name_map = {stock["name"]: stock for stock in MASTER_STOCK_LIST}
        fuzzy_matches = process.extract(query, name_map.keys(), scorer=fuzz.partial_ratio, limit=5)
        for match_name, score, _ in fuzzy_matches:
            stock = name_map[match_name]
            if stock not in results:
                results.append(stock)

    # Return top 5
    return results[:5]


@app.get("/predict")
def predict(symbol: str):
    try:
        result = predict_next_day(symbol)
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


