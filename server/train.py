import os
import torch
import torch.nn as nn
from torch.optim import Adam
import pickle
import numpy as np

from config import MODELS_DIR, DEVICE, LR, EPOCHS_NEW, EPOCHS_UPDATE, SEQ_LENGTH, NORMALIZE_DAYS
from lstm_model import LSTMModel
from utils import fetch_stock_data, normalize_data, create_sequences

def get_model_paths(symbol: str):
    symbol = symbol.upper()
    model_path = os.path.join(MODELS_DIR, f"{symbol}.pt")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")
    return model_path, scaler_path

def train_new_stock(symbol: str, epochs=EPOCHS_NEW):
    prices_raw = fetch_stock_data(symbol)
    prices_to_scale = prices_raw[-NORMALIZE_DAYS:]
    prices_norm, scaler = normalize_data(prices_to_scale)
    X, y = create_sequences(prices_norm)
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    model = LSTMModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    # Save model and scaler
    model_path, scaler_path = get_model_paths(symbol)
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return model

def load_model(symbol: str):
    model_path, scaler_path = get_model_paths(symbol)
    if not os.path.exists(model_path):
        return None, None
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def update_stock_model(symbol: str, epochs=EPOCHS_UPDATE):
    model, scaler = load_model(symbol)
    if model is None:
        return train_new_stock(symbol)
    prices_raw = fetch_stock_data(symbol)
    prices_to_scale = prices_raw[-NORMALIZE_DAYS:]
    prices_norm, _ = normalize_data(prices_to_scale, scaler)
    X, y = create_sequences(prices_norm)
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    model_path, _ = get_model_paths(symbol)
    torch.save(model.state_dict(), model_path)
    return model

def predict_next_day(symbol: str):
    symbol = symbol.upper().strip()
    model, scaler = load_model(symbol)
    if model is None:
        model = train_new_stock(symbol)
        _, scaler = load_model(symbol)
    prices_raw = fetch_stock_data(symbol)
    prices_to_scale = prices_raw[-NORMALIZE_DAYS:]
    prices_norm, _ = normalize_data(prices_to_scale, scaler)
    if len(prices_norm) < SEQ_LENGTH:
        raise ValueError(f"Not enough data for {symbol}")
    last_seq = prices_norm[-SEQ_LENGTH:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq)
    pred = scaler.inverse_transform(pred.cpu().numpy())
    current_price = float(prices_raw[-1][0])
    return {
        "symbol": symbol,
        "current_price": current_price,
        "next_day_prediction": float(pred[0][0])
    }
