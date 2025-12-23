import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from config import SEQ_LENGTH, DATA_DIR

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_stock_data(symbol: str, period="5y", save_csv=True):
    symbol = symbol.upper()
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    df_existing = None
    last_date = None

    # Read existing CSV
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path, header=None)
        df_existing[0] = pd.to_datetime(df_existing[0]).dt.date
        last_date = df_existing[0].max()

    today = pd.Timestamp.utcnow().date()

    # Fetch logic
    if last_date is not None:
        df_new = yf.download(
            symbol,
            start=last_date.strftime("%Y-%m-%d"),
            end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )
    else:
        df_new = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=True
        )

    if df_new.empty and df_existing is None:
        raise ValueError(f"No data found for {symbol}")

    if not df_new.empty:
        df_new = df_new[["Close"]].copy()
        df_new.index = pd.to_datetime(df_new.index).date
        df_new.reset_index(inplace=True)
        df_new.columns = [0, 1]  # date, price

    # Merge
    if df_existing is not None and not df_new.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    elif df_existing is not None:
        df_combined = df_existing
    else:
        df_combined = df_new

    # Clean
    df_combined = df_combined.drop_duplicates(subset=0, keep="last")
    df_combined = df_combined.sort_values(by=0).reset_index(drop=True)

    if save_csv:
        df_combined.to_csv(file_path, header=False, index=False)

    return df_combined[1].values.reshape(-1, 1)



def normalize_data(data, scaler=None):
    """
    Normalize prices to range [0,1].
    Reuse existing scaler for consistency.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data, scaler


def create_sequences(data, seq_length=SEQ_LENGTH):
    """
    Convert time-series into LSTM sequences.
    X: shape (num_samples, seq_length, 1)
    y: shape (num_samples, 1)
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)
