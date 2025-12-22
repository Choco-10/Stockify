from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from train import predict_next_day, update_stock_model

app = FastAPI(
    title="Stock Prediction Engine",
    description="LSTM-based stock price prediction system",
    version="1.0.0"
)

class UpdateRequest(BaseModel):
    symbols: List[str]


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


@app.get("/health")
def health():
    return {"status": "ok"}


