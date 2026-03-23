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
from utils import fetch_stock_data, normalize_data, create_sequences


logger = logging.getLogger(__name__)


def get_model_paths(symbol: str):
    symbol = symbol.upper()
    onnx_path = os.path.join(MODELS_DIR, f"{symbol}.onnx")
    scaler_meta_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.json")
    return onnx_path, scaler_meta_path


def save_scaler_metadata(scaler, scaler_meta_path: str):
    scaler_payload = {
        "data_min": float(scaler.data_min_[0]),
        "data_max": float(scaler.data_max_[0]),
    }
    with open(scaler_meta_path, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f)


def load_scaler_metadata(scaler_meta_path: str):
    if not os.path.exists(scaler_meta_path):
        return None
    with open(scaler_meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "data_min": float(payload["data_min"]),
        "data_max": float(payload["data_max"]),
    }


def normalize_with_metadata(data, scaler_meta):
    data_min = scaler_meta["data_min"]
    data_max = scaler_meta["data_max"]
    denom = data_max - data_min
    if denom == 0:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - data_min) / denom).astype(np.float32)


def inverse_with_metadata(values, scaler_meta):
    data_min = scaler_meta["data_min"]
    data_max = scaler_meta["data_max"]
    denom = data_max - data_min
    if denom == 0:
        return np.full_like(values, fill_value=data_min, dtype=np.float32)
    return (values * denom + data_min).astype(np.float32)


def export_model_to_onnx(model: LSTMModel, onnx_path: str):
    model.eval()
    dummy_input = torch.randn(1, SEQ_LENGTH, 1, dtype=torch.float32).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["inputs"],
        output_names=["prediction"],
        dynamo=False,
    )


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
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    onnx_path, scaler_meta_path = get_model_paths(symbol)
    save_scaler_metadata(scaler, scaler_meta_path)
    try:
        export_model_to_onnx(model, onnx_path)
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

    prices_raw = fetch_stock_data(symbol)
    prices_to_scale = prices_raw[-NORMALIZE_DAYS:]
    prices_norm = normalize_with_metadata(prices_to_scale, scaler_meta)

    if len(prices_norm) < SEQ_LENGTH:
        raise ValueError(f"Not enough data for {symbol}")

    last_seq_np = np.expand_dims(prices_norm[-SEQ_LENGTH:].astype(np.float32), axis=0)
    input_name = onnx_session.get_inputs()[0].name
    pred_norm = onnx_session.run(None, {input_name: last_seq_np})[0]
    pred = inverse_with_metadata(pred_norm, scaler_meta)

    current_price = float(prices_raw[-1][0])
    return {
        "symbol": symbol,
        "current_price": current_price,
        "next_day_prediction": float(pred[0][0]),
    }
