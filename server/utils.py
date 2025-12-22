import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from config import SEQ_LENGTH, DATA_DIR, NORMALIZE_DAYS

# Fetch historical closing prices
def fetch_stock_data(symbol: str, period="5y", save_csv=True):
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    df = df[["Close"]]
    if save_csv:
        file_path = os.path.join(DATA_DIR, f"{symbol.upper()}.csv")
        df.to_csv(file_path)
    return df["Close"].values.reshape(-1, 1)

# Normalize prices
def normalize_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler

# Create sequences for LSTM
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)
