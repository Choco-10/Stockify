import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from config import SEQ_LENGTH, DATA_DIR

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def _normalize_col_name(col):
    # Flatten tuples or multi-index labels into a single string
    if isinstance(col, (tuple, list)):
        s = " ".join([str(x) for x in col if x is not None])
    else:
        s = str(col)
    sl = s.lower()
    if "date" in sl:
        return "Date"
    if "open" in sl and "adj" not in sl:
        return "Open"
    if "high" in sl:
        return "High"
    if "low" in sl:
        return "Low"
    if "adj" in sl and "close" in sl:
        return "Adj Close"
    if "close" in sl:
        return "Close"
    if "volume" in sl:
        return "Volume"
    return s


def fetch_stock_data(symbol: str, period="5y", save_csv=True):
    """
    Fetch historical OHLCV data for `symbol` and return a pandas DataFrame with
    columns: Date, Open, High, Low, Close, Adj Close, Volume.

    This function will cache CSVs in `DATA_DIR` and append new rows when
    possible. It is backward-compatible with older CSVs that contained only
    Date and Close, but will normalize saved CSVs to the full OHLCV format
    when new data is fetched.
    """
    symbol = symbol.upper()
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    df_existing = None
    last_date = None

    # Read existing CSV (try to be flexible about headers)
    if os.path.exists(file_path):
        try:
            df_existing = pd.read_csv(file_path, parse_dates=[0])
            # Normalize column names in case the CSV was written with tuple-like
            # or multiindex column names (e.g. when saving a pandas DataFrame
            # coming from yfinance with multi-level columns).
            df_existing.rename(columns={c: _normalize_col_name(c) for c in df_existing.columns}, inplace=True)
            # Drop duplicate column names (keep first occurrence) to avoid
            # DataFrame slices returning multiple columns for a single name.
            df_existing = df_existing.loc[:, ~df_existing.columns.duplicated()]
            # Legacy two-column CSV: Date, Close
            if df_existing.shape[1] == 2 and set(df_existing.columns) >= {"Date", "Close"}:
                df_existing = df_existing[["Date", "Close"]]

            # Coerce Date values safely and ignore missing entries when computing last_date
            df_existing["Date"] = pd.to_datetime(df_existing["Date"], errors="coerce").dt.date
            last_dates = df_existing["Date"].dropna()
            last_date = last_dates.max() if not last_dates.empty else None
        except Exception:
            df_existing = None
            last_date = None

    today = pd.Timestamp.utcnow().date()

    # Fetch from yfinance
    if last_date is not None:
        df_new = yf.download(
            symbol,
            start=last_date.strftime("%Y-%m-%d"),
            end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
    else:
        df_new = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=False,
        )

    if df_new is None or df_new.empty:
        if df_existing is None:
            raise ValueError(f"No data found for {symbol}")
        else:
            # return existing as DataFrame
            return _ensure_ohlcv_df(df_existing)

    # Keep relevant columns and reset index to a Date column
    df_new = df_new.reset_index()
    # Normalize column names (yfinance often returns MultiIndex columns like ('Close','AAPL'))
    df_new.rename(columns={c: _normalize_col_name(c) for c in df_new.columns}, inplace=True)
    # Drop duplicate column names after normalization
    df_new = df_new.loc[:, ~df_new.columns.duplicated()]
    df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.date
    # Keep only OHLCV columns (if present)
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df_new.columns]
    df_new = df_new[cols].copy()

    # Merge with existing
    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    # Flatten/normalize any tuple-like or MultiIndex column names produced by yfinance
    try:
        new_cols = [_normalize_col_name(c) for c in df_combined.columns]
        df_combined.columns = new_cols
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    except Exception:
        pass
    # Ensure a Date column exists (some CSVs or yfinance outputs use the index
    # or different column names). Try to recover a Date column before dedup.
    if "Date" not in df_combined.columns:
        # If the index is a DatetimeIndex, reset it into a Date column.
        if isinstance(df_combined.index, pd.DatetimeIndex):
            df_combined = df_combined.reset_index()
            # rename the first column (index) to Date if needed
            if df_combined.columns[0] != "Date":
                df_combined.rename(columns={df_combined.columns[0]: "Date"}, inplace=True)
        else:
            # Try to find any column that looks like a date
            for c in list(df_combined.columns):
                if "date" in str(c).lower():
                    df_combined.rename(columns={c: "Date"}, inplace=True)
                    break

    # Clean and sort
    try:
        df_combined = df_combined.drop_duplicates(subset=["Date"], keep="last")
    except Exception:
        # If Date isn't usable, try to reset the index into a Date column then dedup.
        try:
            if isinstance(df_combined.index, pd.DatetimeIndex):
                df_combined = df_combined.reset_index()
                if df_combined.columns[0] != "Date":
                    df_combined.rename(columns={df_combined.columns[0]: "Date"}, inplace=True)
                df_combined = df_combined.drop_duplicates(subset=["Date"], keep="last")
            else:
                # Last-resort: drop duplicate rows entirely
                df_combined = df_combined.drop_duplicates(keep="last")
        except Exception:
            df_combined = df_combined.drop_duplicates(keep="last")
    df_combined = df_combined.sort_values(by="Date").reset_index(drop=True)

    if save_csv:
        # Save with header (modern CSV format)
        df_combined.to_csv(file_path, index=False)

    return _ensure_ohlcv_df(df_combined)


def _ensure_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame that always has these columns:
    Date, Open, High, Low, Close, Adj Close, Volume. Missing columns are filled with Close or zeros.
    """
    df = df.copy()
    # If Date isn't a column, try several heuristics to recover it.
    if "Date" not in df.columns:
        # If index is DatetimeIndex, reset it into a Date column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if df.columns[0] != "Date":
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        else:
            # Look for any column with 'date' in its name (case-insensitive)
            found = False
            for c in list(df.columns):
                if "date" in str(c).lower():
                    df.rename(columns={c: "Date"}, inplace=True)
                    found = True
                    break
            if not found and df.shape[1] >= 1:
                # As a last resort treat the first column as Date
                df = df.reset_index()
                if df.columns[0] != "Date":
                    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Proceed even if Date is not yet present; try to recover it below

    # Fill missing OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in df.columns:
            if col == "Volume":
                df[col] = 0
            else:
                # fallback to Close if available
                df[col] = df.get("Close", 0)

    # Ensure types
    # If Date column still missing, try to recover from the index or first column
    if "Date" not in df.columns:
        try:
            df["Date"] = df.index
        except Exception:
            try:
                if df.shape[1] >= 1:
                    df["Date"] = df.iloc[:, 0]
                else:
                    df["Date"] = pd.NaT
            except Exception:
                df["Date"] = pd.NaT

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    # Drop rows without a valid date (malformed trailing rows)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    # Drop rows with missing numeric data (e.g., today entries with NaN close)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    return df


def normalize_data(data, scaler=None):
    """
    Normalize a 2D array of features to [0,1]. If `scaler` is None a new
    MinMaxScaler is fitted to `data` and returned.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(data)
    else:
        norm = scaler.transform(data)
    return norm, scaler


def create_sequences(data, seq_length=SEQ_LENGTH):
    """
    Convert time-series into LSTM sequences. Works with multi-feature arrays.
    X: shape (num_samples, seq_length, features)
    y: shape (num_samples, features) or (num_samples, 1) depending on input.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def create_return_sequences(prices_raw, prices_norm, seq_length=SEQ_LENGTH):
    """
    Build sequences from normalized multi-feature inputs, with next-step raw
    return target computed from raw Close prices:
      y_return[i] = (P_t - P_{t-1}) / P_{t-1}

    `prices_raw` may be a DataFrame/array of Close prices (shape N or N x 1)
    while `prices_norm` should be a 2D array of normalized features (N x F).
    """
    X, y_return = [], []

    # Extract close prices as 1D numpy array
    if isinstance(prices_raw, pd.DataFrame) or isinstance(prices_raw, pd.Series):
        close_vals = np.asarray(prices_raw["Close"]).reshape(-1)
    else:
        close_vals = np.asarray(prices_raw).reshape(-1)

    for i in range(seq_length, len(close_vals)):
        prev_price = float(close_vals[i - 1])
        curr_price = float(close_vals[i])
        if abs(prev_price) < 1e-8:
            continue
        X.append(prices_norm[i - seq_length:i])
        y_return.append([(curr_price - prev_price) / prev_price])
    return np.array(X), np.array(y_return, dtype=np.float32)
