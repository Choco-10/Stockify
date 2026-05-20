import os
import importlib
import json
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import warnings

try:
    ort = importlib.import_module("onnxruntime")
except Exception:
    ort = None

from config import MODELS_DIR, DEVICE, LR, EPOCHS_NEW, EPOCHS_UPDATE, SEQ_LENGTH, NORMALIZE_DAYS
from lstm_model import LSTMModel
from utils import fetch_stock_data, normalize_data, create_return_sequences


logger = logging.getLogger(__name__)


def get_model_paths(symbol: str):
    symbol = symbol.upper()
    onnx_path = os.path.join(MODELS_DIR, f"{symbol}.onnx")
    scaler_meta_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.json")
    return onnx_path, scaler_meta_path


def save_scaler_metadata(scaler, scaler_meta_path: str):
    # Support multi-feature scalers by saving lists
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    try:
        data_min_list = [float(x) for x in np.asarray(data_min).reshape(-1).tolist()]
        data_max_list = [float(x) for x in np.asarray(data_max).reshape(-1).tolist()]
    except Exception:
        data_min_list = float(np.asarray(data_min).reshape(-1)[0])
        data_max_list = float(np.asarray(data_max).reshape(-1)[0])

    scaler_payload = {
        "data_min": data_min_list,
        "data_max": data_max_list,
        "target_type": "return",
        "num_features": len(data_min_list) if isinstance(data_min_list, list) else 1,
    }
    with open(scaler_meta_path, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f)


def load_scaler_metadata(scaler_meta_path: str):
    if not os.path.exists(scaler_meta_path):
        return None
    with open(scaler_meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "data_min": payload["data_min"],
        "data_max": payload["data_max"],
        "target_type": payload.get("target_type", "price_norm"),
        "num_features": payload.get("num_features", 1),
    }


def normalize_with_metadata(data, scaler_meta):
    data_min = scaler_meta["data_min"]
    data_max = scaler_meta["data_max"]
    arr = np.asarray(data, dtype=np.float32)
    # Handle scalar (legacy) and list (multi-feature)
    if isinstance(data_min, list) or isinstance(data_max, list):
        mins = np.asarray(data_min, dtype=np.float32).reshape(1, -1)
        maxs = np.asarray(data_max, dtype=np.float32).reshape(1, -1)
        denom = (maxs - mins)
        denom[denom == 0] = 1.0
        return ((arr - mins) / denom).astype(np.float32)
    else:
        denom = float(data_max) - float(data_min)
        if denom == 0:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - float(data_min)) / denom).astype(np.float32)


def inverse_with_metadata(values, scaler_meta):
    data_min = scaler_meta["data_min"]
    data_max = scaler_meta["data_max"]
    arr = np.asarray(values, dtype=np.float32)
    if isinstance(data_min, list) or isinstance(data_max, list):
        mins = np.asarray(data_min, dtype=np.float32).reshape(1, -1)
        maxs = np.asarray(data_max, dtype=np.float32).reshape(1, -1)
        denom = (maxs - mins)
        denom[denom == 0] = 1.0
        return (arr * denom + mins).astype(np.float32)
    else:
        denom = float(data_max) - float(data_min)
        if denom == 0:
            return np.full_like(arr, fill_value=float(data_min), dtype=np.float32)
        return (arr * denom + float(data_min)).astype(np.float32)


def export_model_to_onnx(model: LSTMModel, onnx_path: str, num_features: int = 1):
    model.eval()
    # Ensure export happens on CPU to avoid device-specific artifacts
    cpu_model = model.to("cpu")
    dummy_input = torch.randn(1, SEQ_LENGTH, num_features, dtype=torch.float32).cpu()
    try:
        torch.onnx.export(
            cpu_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=False,
            input_names=["inputs"],
            output_names=["prediction"],
            dynamo=False,
        )
    finally:
        # move model back to original device
        model.to(DEVICE)


def create_onnx_session(onnx_path: str):
    if ort is None or not os.path.exists(onnx_path):
        return None

    available = ort.get_available_providers()
    gpu_priority = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
    ]

    # Auto-use GPU if a supported provider is available and can initialize.
    for provider in gpu_priority:
        if provider not in available:
            continue
        try:
            session = ort.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])
            logger.info("Using ONNX Runtime provider: %s", provider)
            return session
        except Exception as e:
            logger.warning("Failed to initialize provider %s (%s). Trying next provider.", provider, e)

    logger.info("Using ONNX Runtime provider: CPUExecutionProvider")
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def train_new_stock(symbol: str, epochs=EPOCHS_NEW):
    # prices_raw is now a DataFrame with OHLCV columns
    prices_df = fetch_stock_data(symbol)
    # select feature columns to use for training
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    features = prices_df[feature_cols].values.astype(np.float32)

    # Fit scaler on the last NORMALIZE_DAYS rows and apply to full history
    fit_window = features[-NORMALIZE_DAYS:]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(fit_window)
    prices_norm = scaler.transform(features)

    # use raw close prices for computing returns
    close_raw = prices_df[["Close"]].values
    X, y = create_return_sequences(close_raw, prices_norm)

    if len(X) == 0:
        raise ValueError(f"Not enough valid return samples for {symbol}")

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    num_features = prices_norm.shape[1]
    model = LSTMModel(input_size=num_features).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    onnx_path, scaler_meta_path = get_model_paths(symbol)
    save_scaler_metadata(scaler, scaler_meta_path)
    try:
        export_model_to_onnx(model, onnx_path, num_features=num_features)
    except Exception as e:
        warnings.warn(f"ONNX export failed for {symbol}: {e}")


def load_model(symbol: str):
    onnx_path, scaler_meta_path = get_model_paths(symbol)
    scaler_meta = load_scaler_metadata(scaler_meta_path)
    onnx_session = create_onnx_session(onnx_path)
    if scaler_meta is None or onnx_session is None:
        return None, None
    return scaler_meta, onnx_session


def update_stock_model(symbol: str, epochs=EPOCHS_UPDATE):
    # ONNX model artifacts are inference-only, so update by retraining and re-exporting.
    return train_new_stock(symbol, epochs=epochs)


def predict_next_day(symbol: str):
    symbol = symbol.upper().strip()
    scaler_meta, onnx_session = load_model(symbol)

    if scaler_meta is None or onnx_session is None:
        train_new_stock(symbol)
        scaler_meta, onnx_session = load_model(symbol)

    if scaler_meta is None or onnx_session is None:
        raise RuntimeError(f"Model artifacts are missing for {symbol}")

    prices_df = fetch_stock_data(symbol)
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    features = prices_df[feature_cols].values.astype(np.float32)
    prices_norm = normalize_with_metadata(features[-NORMALIZE_DAYS:], scaler_meta)

    # Build full normalized sequence for inference according to scaler metadata.
    model_num_features = int(scaler_meta.get("num_features", 1))

    if model_num_features == 1:
        # Old models expect a single feature (Close). Normalize Close using metadata.
        close_idx = feature_cols.index("Close") if "Close" in feature_cols else 0
        close_vals = features[:, close_idx].reshape(-1, 1)
        prices_norm_full = normalize_with_metadata(close_vals, scaler_meta)
    else:
        # Multi-feature model: ensure scaler metadata has matching feature counts
        if isinstance(scaler_meta.get("data_min"), list) and len(scaler_meta.get("data_min")) == features.shape[1]:
            mins = np.asarray(scaler_meta["data_min"]).reshape(1, -1)
            maxs = np.asarray(scaler_meta["data_max"]).reshape(1, -1)
            denom = (maxs - mins)
            denom[denom == 0] = 1.0
            prices_norm_full = ((features - mins) / denom).astype(np.float32)
        else:
            raise RuntimeError(f"Model expects {model_num_features} features but data has {features.shape[1]}")

    if len(prices_norm_full) < SEQ_LENGTH:
        raise ValueError(f"Not enough data for {symbol}")

    last_seq_np = np.expand_dims(prices_norm_full[-SEQ_LENGTH:].astype(np.float32), axis=0)
    input_name = onnx_session.get_inputs()[0].name
    pred_out = onnx_session.run(None, {input_name: last_seq_np})[0]

    current_price = float(prices_df["Close"].values[-1])

    target_type = scaler_meta.get("target_type", "price_norm")
    if target_type == "return":
        pred_return = float(pred_out[0][0])
        next_day_prediction = float(current_price * (1.0 + pred_return))
    else:
        # Backward compatibility for previously exported price-normalized models.
        pred_price = inverse_with_metadata(pred_out, scaler_meta)
        next_day_prediction = float(pred_price[0][0])

    return {
        "symbol": symbol,
        "current_price": current_price,
        "next_day_prediction": next_day_prediction,
    }
