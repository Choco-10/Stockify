import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from config import SEQ_LENGTH, DATA_DIR, YF_RETRY_ATTEMPTS, YF_RETRY_BACKOFF

logger = logging.getLogger(__name__)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# Version tag for the feature schema. Bump when columns change so the
# per-stock cache transparently invalidates.
_TECH_FEATURE_TAG = "v2_trend_regime"


def get_cached_technical_features(symbol: str, prices_df: pd.DataFrame, force_recompute: bool = False) -> np.ndarray:
    """
    Return the technical-feature array for `symbol`, computing and caching it on miss.

    The returned array contains, in order:
      1. Base OHLCV (5 cols)
      2. Existing technical indicators (RSI, BB %B, vol ratio, etc.) (12 cols)
      3. NEW trend features (sma50/200 ratios, slope_20/50, volume_mom) (5 cols)
      4. NEW market regime features (SPY/QQQ 1d/5d/20d returns) (6 cols) —
         only included when MARKET_REGIME_ENABLED is True and the
         per-symbol regime cache can be built.

    The cache key is (symbol, last Date in prices_df, CSV mtime, schema tag)
    so any data refresh OR a schema bump transparently invalidates the cache.

    This is the single highest-leverage optimization: ``add_technical_features`` runs
    10+ pandas rolling/ewm calls and is otherwise re-executed for every fit *and*
    every predict call (and per ensemble member).
    """
    from config import MARKET_REGIME_ENABLED  # local import to avoid cycle

    symbol = symbol.upper()
    cache_npy = os.path.join(DATA_DIR, f"{symbol}_tech.npy")
    cache_meta = os.path.join(DATA_DIR, f"{symbol}_tech.meta.json")
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    # Build a stable cache key. The schema tag is appended so the new
    # trend/regime columns trigger a recompute exactly once on deploy.
    last_date = ""
    if prices_df is not None and not prices_df.empty and "Date" in prices_df.columns:
        try:
            last_dt = pd.to_datetime(prices_df["Date"], errors="coerce").max()
            if pd.notna(last_dt):
                last_date = last_dt.strftime("%Y-%m-%d")
        except Exception:
            last_date = ""
    try:
        csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0.0
    except OSError:
        csv_mtime = 0.0
    raw_key = (
        f"{symbol}|{last_date}|{csv_mtime:.6f}|"
        f"{len(prices_df) if prices_df is not None else 0}|"
        f"{_TECH_FEATURE_TAG}|regime={int(bool(MARKET_REGIME_ENABLED))}"
    )
    key_hash = hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    # Try cache hit.
    if not force_recompute and os.path.exists(cache_npy) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("cache_key") == key_hash:
                arr = np.load(cache_npy)
                logger.debug("Loaded cached technical features for %s (%d rows, %d cols)",
                             symbol, arr.shape[0], arr.shape[1])
                return arr
        except Exception as e:
            logger.warning("Cache read failed for %s, recomputing: %s", symbol, e)

    # Miss: compute base+trend, then append market regime if enabled.
    features = add_technical_features(prices_df, include_trend=True)
    if MARKET_REGIME_ENABLED and prices_df is not None and "Date" in prices_df.columns:
        try:
            regime = get_cached_market_regime_features(symbol, prices_df["Date"])
            if regime is not None and regime.shape[0] == features.shape[0]:
                features = np.column_stack([features, regime.astype(np.float32)])
            else:
                logger.warning(
                    "Regime shape %s does not match features %s for %s; skipping.",
                    None if regime is None else regime.shape,
                    features.shape, symbol,
                )
        except Exception as e:
            logger.warning("Market regime features unavailable for %s: %s", symbol, e)
    # Replace any NaN/Inf left over with 0.5 (neutral).
    features = np.nan_to_num(features, nan=0.5, posinf=1.0, neginf=0.0)
    try:
        np.save(cache_npy, features)
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({"cache_key": key_hash, "symbol": symbol,
                       "last_date": last_date, "rows": int(features.shape[0]),
                       "cols": int(features.shape[1])}, f)
        logger.info("Cached technical features for %s -> %s (%d cols)",
                    symbol, cache_npy, features.shape[1])
    except Exception as e:
        # Cache failures must never break training/inference.
        logger.warning("Cache write failed for %s: %s", symbol, e)
    return features


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

    Caches CSVs in `DATA_DIR` and appends new rows when possible.
    Backward-compatible with older CSVs that contained only Date and Close.
    """
    symbol = symbol.upper()
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    df_existing = None
    last_date = None

    # Read existing CSV (try to be flexible about headers)
    if os.path.exists(file_path):
        try:
            df_existing = pd.read_csv(file_path, parse_dates=[0])
            # Normalize column names
            df_existing.rename(columns={c: _normalize_col_name(c) for c in df_existing.columns}, inplace=True)
            # Drop duplicate column names (keep first occurrence)
            df_existing = df_existing.loc[:, ~df_existing.columns.duplicated()]
            # Legacy two-column CSV: Date, Close
            if df_existing.shape[1] == 2 and set(df_existing.columns) >= {"Date", "Close"}:
                df_existing = df_existing[["Date", "Close"]]

            # Coerce Date values safely and ignore missing entries when computing last_date
            df_existing["Date"] = pd.to_datetime(df_existing["Date"], errors="coerce").dt.date
            last_dates = df_existing["Date"].dropna()
            last_date = last_dates.max() if not last_dates.empty else None
        except Exception as e:
            logger.warning("Failed to read existing CSV for %s: %s", symbol, e)
            df_existing = None
            last_date = None

    today = pd.Timestamp.utcnow().date()

    # Fetch from yfinance with retry + exponential backoff
    df_new = None
    for attempt in range(YF_RETRY_ATTEMPTS):
        try:
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
            break  # success — exit retry loop
        except Exception as e:
            wait = YF_RETRY_BACKOFF * (2 ** attempt)
            if attempt < YF_RETRY_ATTEMPTS - 1:
                logger.warning(
                    "yfinance download failed for %s (attempt %d/%d): %s — retrying in %.0fs",
                    symbol, attempt + 1, YF_RETRY_ATTEMPTS, e, wait,
                )
                import time as _time
                _time.sleep(wait)
            else:
                logger.error(
                    "yfinance download failed for %s after %d attempts: %s",
                    symbol, YF_RETRY_ATTEMPTS, e,
                )
                raise

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
    # Ensure a Date column exists
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
    """Return a DataFrame that always has columns:
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


def create_return_sequences(prices_raw, prices_norm, seq_length=SEQ_LENGTH):
    """
    Build sequences from normalized multi-feature inputs, with next-step raw
    return target computed from raw Close prices:
      y_return[i] = (P_t - P_{t-1}) / P_{t-1}

    `prices_raw` may be a DataFrame/array of Close prices (shape N or N x 1)
    while `prices_norm` should be a 2D array of normalized features (N x F).

    Returns:
        X:            input sequences [N, seq_length, n_features]
        y_return:     target returns  [N, 1]
        y_direction:  target directions [N, 1]  (1.0 = up, 0.0 = down)
    """
    X, y_return, y_direction = [], [], []

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
        ret = (curr_price - prev_price) / prev_price
        y_return.append([ret])
        y_direction.append([1.0 if ret >= 0 else 0.0])
    return (np.array(X),
            np.array(y_return, dtype=np.float32),
            np.array(y_direction, dtype=np.float32))


# ────────────────────────────────────────────────────────────────────────────
# Technical Indicators as Additional Features
# ────────────────────────────────────────────────────────────────────────────
# Pure-numpy implementations of the rolling/ewm primitives used below. These
# replace the previous pandas rolling()/ewm() chain (2-10x faster, no
# DataFrame/Series metadata, and friendly to numba in the future).

def _rolling_mean_min1(x, w):
    """Rolling mean with min_periods=1. First (w-1) entries are expanding means."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        s += x[i]
        if i < w:
            out[i] = s / (i + 1)
        else:
            s -= x[i - w]
            out[i] = s / w
    return out


def _rolling_std_min1(x, w, ddof=0):
    """Rolling std with min_periods=1 and configurable ddof."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < w:
            window = x[:i + 1]
        else:
            window = x[i - w + 1:i + 1]
        out[i] = np.std(window, ddof=ddof) if len(window) > 0 else 0.0
    return out


def _rolling_max_min1(x, w):
    """Rolling max with min_periods=1. First (w-1) entries are expanding max."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    cur = -np.inf
    for i in range(n):
        if i < w:
            cur = x[i] if cur == -np.inf else max(cur, x[i])
            out[i] = cur
        else:
            out[i] = x[i - w + 1:i + 1].max()
    return out


def _rolling_min_min1(x, w):
    """Rolling min with min_periods=1. First (w-1) entries are expanding min."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    cur = np.inf
    for i in range(n):
        if i < w:
            cur = x[i] if cur == np.inf else min(cur, x[i])
            out[i] = cur
        else:
            out[i] = x[i - w + 1:i + 1].min()
    return out


def _ewm(x, span):
    """Vectorized EMA: y[i] = alpha*x[i] + (1-alpha)*y[i-1], alpha = 2/(span+1)."""
    alpha = 2.0 / (span + 1.0)
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    out[0] = x[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * x[i] + one_minus_alpha * out[i - 1]
    return out


def _pct_change(x, periods=1):
    """Returns array of same length as x; first `periods` values are 0.
    Division by ~0 is avoided by replacing with the input itself in the
    numerator and denominator (yields 0, matching pandas' NaN-fill behavior)."""
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if periods >= n:
        return out
    prev = x[:-periods]
    safe_prev = np.where(np.abs(prev) < 1e-8, 1.0, prev)
    out[periods:] = (x[periods:] - prev) / safe_prev
    return out


def _rolling_slope(x, w):
    """
    Rolling OLS slope of `x` over a window of size `w`.

    Computed closed-form per index i:
        slope_i = (n * sum(t*y) - sum(t)*sum(y)) /
                  (n * sum(t^2) - (sum(t))^2)
    where t = [0, 1, ..., n-1] and y is the windowed slice of x.

    Returns a length-`len(x)` array; entries before the window is full use
    the expanding-window slope. Result is then divided by the local std
    (price-volatility-normalized) to make slopes scale-invariant; the
    caller does the final clipping/squashing.

    Output unit: "fraction of price per bar" (e.g. +0.01 = +1% per bar).
    """
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    # Pre-compute the constant sum_t, sum_t2 for window size w and the
    # expanding versions for warm-up.
    t_full = np.arange(w, dtype=np.float64)
    sum_t_full = t_full.sum()
    sum_t2_full = (t_full * t_full).sum()
    for i in range(n):
        if i < w:
            length = i + 1
            t = np.arange(length, dtype=np.float64)
            y = x[: length]
        else:
            length = w
            t = t_full
            y = x[i - w + 1: i + 1]
        sum_t = sum_t_full if length == w else t.sum()
        sum_t2 = sum_t2_full if length == w else (t * t).sum()
        sum_y = y.sum()
        sum_ty = (t * y).sum()
        denom = length * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-12:
            out[i] = 0.0
            continue
        slope = (length * sum_ty - sum_t * sum_y) / denom
        # Scale-invariant: divide by the local std so a steep slope on a
        # volatile stock is comparable to a gentle slope on a quiet one.
        # Windowed std is computed on the same y slice.
        local_std = float(np.std(y)) if length > 1 else 0.0
        if local_std < 1e-12:
            out[i] = 0.0
        else:
            out[i] = slope / local_std
    return out


def add_technical_features(df, include_trend=True):

    """
    Augment OHLCV DataFrame with enhanced technical indicator features.

    Adds: RSI(14), Bollinger %B, Volume Ratio, Price Position,
          Close/SMA Ratio, Volatility, MACD, ATR, Lagged Returns.

    All rolling/ewm primitives are pure-numpy (see helpers above) which is
    2-10x faster than the previous pandas rolling chain.

    Returns a 2D numpy array where each row is a feature vector
    with all values normalized to approximately [0, 1].
    """
    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    volume = df['Volume'].values.astype(np.float64)

    n = len(df)

    # ── RSI(14) ──────────────────────────────────────────────────────
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    diff[1:] = close[1:] - close[:-1]
    gain = np.clip(diff, 0.0, None)
    loss = np.clip(-diff, 0.0, None)
    avg_gain = _rolling_mean_min1(gain, 14)
    avg_loss = _rolling_mean_min1(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_norm = np.clip(rsi / 100.0, 0.0, 1.0)
    rsi_norm = np.nan_to_num(rsi_norm, nan=0.5)

    # ── Bollinger %B ─────────────────────────────────────────────────
    sma20 = _rolling_mean_min1(close, 20)
    std20 = _rolling_std_min1(close, 20, ddof=0)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    bb_pct_b = (close - lower) / (upper - lower + 1e-10)
    bb_pct_b = np.clip(bb_pct_b, 0.0, 1.0)
    bb_pct_b = np.nan_to_num(bb_pct_b, nan=0.5)

    # ── Volume Ratio (relative to 20-day average) ────────────────────
    vol_sma20 = _rolling_mean_min1(volume, 20)
    vol_ratio = volume / (vol_sma20 + 1e-10)
    vol_ratio = np.clip(vol_ratio, 0.0, 3.0) / 3.0
    vol_ratio = np.nan_to_num(vol_ratio, nan=0.5)

    # ── Price Position within recent range ──────────────────────────
    rolling_high = _rolling_max_min1(high, 20)
    rolling_low = _rolling_min_min1(low, 20)
    price_position = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)
    price_position = np.nan_to_num(price_position, nan=0.5)

    # ── Close / SMA Ratio ─────────────────────────────────────────
    close_sma_ratio = close / (sma20 + 1e-10)
    close_sma_ratio = np.clip(close_sma_ratio, 0.5, 1.5) - 0.5  # shift to [0, 1]
    close_sma_ratio = np.nan_to_num(close_sma_ratio, nan=0.5)

    # ── Volatility (20-day rolling std of returns) ──────────────────
    ret_1d_raw = _pct_change(close, periods=1)
    vola = _rolling_std_min1(ret_1d_raw, 20, ddof=0)
    vola = np.clip(vola * 100.0, 0.0, 5.0) / 5.0
    vola = np.nan_to_num(vola, nan=0.5)

    # ── MACD (Moving Average Convergence Divergence) ───────────────────
    ema12 = _ewm(close, 12)
    ema26 = _ewm(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ewm(macd_line, 9)
    macd_hist = macd_line - signal_line  # histogram

    macd_hist_norm = macd_hist / (close + 1e-10)
    macd_hist_norm = np.clip(macd_hist_norm, -0.05, 0.05) / 0.05  # -> [-1, 1]
    macd_hist_norm = (macd_hist_norm + 1.0) / 2.0  # -> [0, 1]
    macd_hist_norm = np.nan_to_num(macd_hist_norm, nan=0.5)

    # MACD line minus signal line sign (1 if bullish, 0 if bearish)
    macd_sign = (macd_hist >= 0).astype(np.float64)
    macd_sign = np.nan_to_num(macd_sign, nan=0.5)

    # ── ATR (Average True Range) ──────────────────────────────────────
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_series = _rolling_mean_min1(true_range, 14)

    # Normalize ATR by close price (relative volatility)
    atr_norm = atr_series / (close + 1e-10)
    atr_norm = np.clip(atr_norm, 0.0, 0.1) / 0.1  # -> [0, 1]
    atr_norm = np.nan_to_num(atr_norm, nan=0.5)

    # ── Lagged Returns (1-day, 5-day, 20-day) ────────────────────────────
    ret_5d_raw = _pct_change(close, periods=5)
    ret_20d_raw = _pct_change(close, periods=20)

    # Normalize returns to [0, 1] via tanh scaling
    ret_1d_norm = (np.tanh(ret_1d_raw * 50.0) + 1.0) / 2.0   # capture ~±2% moves
    ret_5d_norm = (np.tanh(ret_5d_raw * 10.0) + 1.0) / 2.0    # capture ~±10% moves
    ret_20d_norm = (np.tanh(ret_20d_raw * 5.0) + 1.0) / 2.0   # capture ~±20% moves

    ret_1d_norm = np.nan_to_num(ret_1d_norm, nan=0.5)
    ret_5d_norm = np.nan_to_num(ret_5d_norm, nan=0.5)
    ret_20d_norm = np.nan_to_num(ret_20d_norm, nan=0.5)

    # ── Trend features (SMA ratios, slopes, volume momentum) ─────────
    # These are the new "broader-trend" features that complement the
    # short-window RSI / MACD / Bollinger stack above.
    if include_trend:
        # SMA ratios: close / sma_k, tanh-squashed to [0, 1].
        # tanh(3x) saturates near x=±0.5 -> ~±15% deviation from SMA.
        sma50 = _rolling_mean_min1(close, 50)
        sma200 = _rolling_mean_min1(close, 200)
        close_sma50_ratio = np.tanh((close / (sma50 + 1e-10) - 1.0) * 3.0)
        close_sma50_ratio = (close_sma50_ratio + 1.0) / 2.0
        close_sma50_ratio = np.nan_to_num(close_sma50_ratio, nan=0.5)

        close_sma200_ratio = np.tanh((close / (sma200 + 1e-10) - 1.0) * 3.0)
        close_sma200_ratio = (close_sma200_ratio + 1.0) / 2.0
        close_sma200_ratio = np.nan_to_num(close_sma200_ratio, nan=0.5)

        # Trend slopes (scale-invariant).  _rolling_slope returns a
        # std-normalized slope; we tanh-squash and shift to [0, 1].
        slope_20 = np.tanh(_rolling_slope(close, 20) * 5.0)
        slope_20 = (slope_20 + 1.0) / 2.0
        slope_20 = np.nan_to_num(slope_20, nan=0.5)

        slope_50 = np.tanh(_rolling_slope(close, 50) * 5.0)
        slope_50 = (slope_50 + 1.0) / 2.0
        slope_50 = np.nan_to_num(slope_50, nan=0.5)

        # Volume momentum: volume / volume_sma50, clipped to [0, 3], rescaled.
        vol_sma50 = _rolling_mean_min1(volume, 50)
        volume_momentum = volume / (vol_sma50 + 1e-10)
        volume_momentum = np.clip(volume_momentum, 0.0, 3.0) / 3.0
        volume_momentum = np.nan_to_num(volume_momentum, nan=0.5)

        trend_features = np.column_stack([
            close_sma50_ratio,
            close_sma200_ratio,
            slope_20,
            slope_50,
            volume_momentum,
        ]).astype(np.float32)
    else:
        trend_features = np.empty((n, 0), dtype=np.float32)

    # ── Stack all features ─────────────────────────────────────
    base_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
    tech_features = np.column_stack([
        rsi_norm,
        bb_pct_b,
        vol_ratio,
        price_position,
        close_sma_ratio,
        vola,
        macd_hist_norm,
        macd_sign,
        atr_norm,
        ret_1d_norm,
        ret_5d_norm,
        ret_20d_norm,
    ]).astype(np.float32)

    # Replace any NaN/Inf left over with 0.5 (neutral value after normalization)
    tech_features = np.nan_to_num(tech_features, nan=0.5, posinf=1.0, neginf=0.0)

    return np.column_stack([base_features, tech_features, trend_features])


# ────────────────────────────────────────────────────────────────────────────
# Data Augmentation (Noise Injection)
# ────────────────────────────────────────────────────────────────────────────

def augment_sequences(X, y_returns, y_direction=None,
                      noise_std=0.01, scale_range=(0.95, 1.05)):
    """
    Augment training data with noise injection and scaling.

    Returns augmented X, y_returns, and y_direction (if provided).
    """
    if y_direction is not None:
        X_aug, y_aug, d_aug = [], [], []
    else:
        X_aug, y_aug = [], []

    for i in range(len(X)):
        # Original
        X_aug.append(X[i])
        y_aug.append(y_returns[i])
        if y_direction is not None:
            d_aug.append(y_direction[i])

        # Gaussian noise
        X_noisy = X[i] + np.random.normal(0, noise_std, X[i].shape).astype(np.float32)
        X_aug.append(X_noisy)
        y_aug.append(y_returns[i])
        if y_direction is not None:
            d_aug.append(y_direction[i])

        # Scale input slightly (proportional target)
        scale = np.random.uniform(*scale_range)
        X_scaled = X[i] * scale
        y_scaled = y_returns[i] * scale
        X_aug.append(X_scaled.astype(np.float32))
        y_aug.append(y_scaled.astype(np.float32))
        if y_direction is not None:
            # Direction remains the same (since target scaled proportionally)
            d_aug.append(y_direction[i])

    result = [np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.float32)]
    if y_direction is not None:
        result.append(np.array(d_aug, dtype=np.float32))
    return tuple(result)


# ────────────────────────────────────────────────────────────────────────────
# Market Regime Features (SPY / QQQ)
# ────────────────────────────────────────────────────────────────────────────
# These are appended to the per-stock feature matrix to give the LSTM
# broad market context. The CSVs are cached exactly like stock CSVs
# (see `fetch_stock_data`); the *aligned* feature array is computed once
# per (stock, regime_symbol) pair and cached to <DATA_DIR>/<STOCK>_regime_v1.npy.

# A versioned tag so we can invalidate the regime cache in lock-step with
# the rest of the feature cache. Bump this when the regime features change.
_REGIME_TAG = "v1"


def _compute_regime_return_array(benchmark_df: pd.DataFrame) -> np.ndarray:
    """
    Given a benchmark OHLCV DataFrame (Date-indexed), return a (N, 3) float32
    array of tanh-normalized 1d / 5d / 20d returns:

        ret_1d_norm  = (tanh(ret_1d  * 50) + 1) / 2     (capture ~±2%)
        ret_5d_norm  = (tanh(ret_5d  * 10) + 1) / 2     (capture ~±10%)
        ret_20d_norm = (tanh(ret_20d *  5) + 1) / 2     (capture ~±20%)

    Matches the scaling used for the stock's own lagged returns so the
    network sees regime returns on the same scale as self-returns.
    """
    close = benchmark_df["Close"].values.astype(np.float64)
    ret_1d_raw = _pct_change(close, periods=1)
    ret_5d_raw = _pct_change(close, periods=5)
    ret_20d_raw = _pct_change(close, periods=20)

    ret_1d_norm = (np.tanh(ret_1d_raw * 50.0) + 1.0) / 2.0
    ret_5d_norm = (np.tanh(ret_5d_raw * 10.0) + 1.0) / 2.0
    ret_20d_norm = (np.tanh(ret_20d_raw * 5.0) + 1.0) / 2.0

    out = np.column_stack([ret_1d_norm, ret_5d_norm, ret_20d_norm]).astype(np.float32)
    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)


def get_cached_market_regime_features(
    stock_symbol: str,
    stock_dates: pd.Series,
    symbols=None,
) -> np.ndarray:
    """
    Return a (N_stock, 6) float32 array of market-regime features for `stock_symbol`,
    aligned to the stock's `stock_dates`. Columns:

        [SPY ret_1d, SPY ret_5d, SPY ret_20d,
         QQQ ret_1d, QQQ ret_5d, QQQ ret_20d]

    Cached per stock to <DATA_DIR>/<STOCK>_regime_<tag>.npy. The CSVs for the
    benchmark symbols are stored alongside the stock CSVs (re-using
    `fetch_stock_data`).

    On any benchmark fetch / cache failure, returns a neutral all-0.5 array
    so the model still trains without market context.
    """
    from config import MARKET_REGIME_SYMBOLS  # local import to avoid cycle
    if symbols is None:
        symbols = MARKET_REGIME_SYMBOLS

    stock_symbol = stock_symbol.upper()
    n = len(stock_dates)
    out = np.full((n, 6), 0.5, dtype=np.float32)  # neutral fallback

    # Normalize stock dates to a pandas datetime Series for alignment.
    try:
        dates_norm = pd.to_datetime(pd.Series(stock_dates).values, errors="coerce")
    except Exception:
        return out

    cache_npy = os.path.join(DATA_DIR, f"{stock_symbol}_regime_{_REGIME_TAG}.npy")
    cache_meta = os.path.join(DATA_DIR, f"{stock_symbol}_regime_{_REGIME_TAG}.meta.json")

    # Build a stable cache key from stock-dates span + benchmark CSV mtims.
    try:
        dates_min = "" if pd.isna(dates_norm.min()) else pd.Timestamp(dates_norm.min()).strftime("%Y-%m-%d")
        dates_max = "" if pd.isna(dates_norm.max()) else pd.Timestamp(dates_norm.max()).strftime("%Y-%m-%d")
    except Exception:
        dates_min, dates_max = "", ""
    benchmark_mtimes = []
    for s in symbols:
        bp = os.path.join(DATA_DIR, f"{s}.csv")
        try:
            benchmark_mtimes.append(f"{s}:{os.path.getmtime(bp):.6f}")
        except OSError:
            benchmark_mtimes.append(f"{s}:NA")
    raw_key = f"{stock_symbol}|{dates_min}|{dates_max}|{n}|" + ",".join(benchmark_mtimes)
    key_hash = hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    # Try cache hit.
    if os.path.exists(cache_npy) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("cache_key") == key_hash:
                arr = np.load(cache_npy)
                if arr.shape == (n, 6):
                    logger.debug("Loaded cached regime features for %s (%d rows)",
                                 stock_symbol, arr.shape[0])
                    return arr.astype(np.float32)
        except Exception as e:
            logger.warning("Regime cache read failed for %s, recomputing: %s",
                           stock_symbol, e)

    # Miss: fetch each benchmark, compute returns, align to stock dates.
    col_offset = 0
    for sym in symbols:
        try:
            bench_df = fetch_stock_data(sym)
        except Exception as e:
            logger.warning("Failed to fetch benchmark %s for %s: %s",
                           sym, stock_symbol, e)
            col_offset += 3
            continue
        # Compute the (M, 3) return array on the benchmark's own timeline.
        bench_features = _compute_regime_return_array(bench_df)
        bench_dates = pd.to_datetime(bench_df["Date"], errors="coerce")
        # Align to stock dates: for each stock date, find the most recent
        # benchmark date at-or-before it. If none, leave neutral (0.5).
        try:
            bench_index = pd.DataFrame(
                {"date": bench_dates.values, "row": np.arange(len(bench_dates))}
            ).dropna(subset=["date"]).sort_values("date")
            # For each stock date, find rightmost bench row <= stock date.
            merged = pd.merge_asof(
                pd.DataFrame({"date": dates_norm}).sort_values("date"),
                bench_index,
                on="date",
                direction="backward",
            )
            bench_rows = merged["row"].fillna(-1).astype(int).to_numpy()
            aligned = np.where(
                (bench_rows >= 0)[:, None],
                bench_features[np.clip(bench_rows, 0, len(bench_features) - 1)],
                0.5,
            )
            # Re-order to match the original stock_dates order.
            sort_perm = pd.DataFrame({"date": dates_norm}).sort_values("date") \
                .index.to_numpy()
            # merged is already sorted by date; undo the sort to original order.
            inv = np.empty_like(sort_perm)
            inv[sort_perm] = np.arange(len(sort_perm))
            aligned = aligned[inv]
            out[:, col_offset:col_offset + 3] = aligned.astype(np.float32)
        except Exception as e:
            logger.warning("Failed to align %s regime to %s: %s", sym, stock_symbol, e)
        col_offset += 3

    # Persist (best-effort).
    try:
        np.save(cache_npy, out)
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({"cache_key": key_hash, "symbol": stock_symbol,
                       "tag": _REGIME_TAG, "rows": int(out.shape[0]),
                       "cols": int(out.shape[1])}, f)
    except Exception as e:
        logger.warning("Regime cache write failed for %s: %s", stock_symbol, e)
    return out
