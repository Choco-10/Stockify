import os
import importlib
import json
import logging
import re
import time
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import warnings
from tqdm import tqdm

try:
    ort = importlib.import_module("onnxruntime")
except Exception:
    ort = None

from config import (
    MODELS_DIR, DEVICE, LR, EPOCHS_NEW,
    SEQ_LENGTH, NORMALIZE_DAYS, HIDDEN_SIZE, NUM_LAYERS,
    GRAD_CLIP_NORM,
    SCHEDULER_T_0, SCHEDULER_T_MULT, SCHEDULER_ETA_MIN,
    ENSEMBLE_CONFIGS, BATCH_SIZE,
    ENABLE_AMP, ENABLE_TORCH_COMPILE,
    CONFIDENCE_TEMPERATURE,
)
from lstm_model import LSTMModel, multi_task_loss
from utils import (
    fetch_stock_data, create_return_sequences,
    get_cached_technical_features,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA / AMP setup (cuDNN benchmark, autocast, GradScaler)
# ---------------------------------------------------------------------------
# Sequence lengths are fixed per model, so cuDNN's benchmark mode is safe
# and picks the fastest convolution algorithm on the first batch of each
# new shape. This typically gives a small but free speedup.

_USE_CUDA = torch.cuda.is_available()
if _USE_CUDA:
    torch.backends.cudnn.benchmark = True

# PyTorch >= 2.0 moved AMP under torch.amp while still exporting
# torch.cuda.amp.* for backward compatibility. We try both, then fall back
# to a no-op context manager on CPU / older versions.
#
# Detection: the new `torch.amp.autocast` signature is
#     autocast(device_type, ...)
# while the legacy `torch.cuda.amp.autocast` signature is
#     autocast()
# We probe the actual signature to pick the right path.
import inspect as _inspect
_AMP_DEVICE_KW = {}
try:
    from torch.amp import autocast as _amp_autocast
    from torch.amp import GradScaler as _GradScaler
    # New API takes device_type as a positional arg; pass it that way.
    if "device_type" in _inspect.signature(_amp_autocast).parameters:
        _AMP_DEVICE_KW = {"device_type": "cuda"}
    else:
        # Older torch.amp.autocast may still need device=
        _AMP_DEVICE_KW = {"device": "cuda"}
except Exception:
    try:
        from torch.cuda.amp import autocast as _amp_autocast  # type: ignore
        from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
        _AMP_DEVICE_KW = {}
    except Exception:
        _amp_autocast = None
        _GradScaler = None
        _AMP_DEVICE_KW = {}


def _autocast_ctx(enabled: bool):
    """Return an autocast context (no-op when AMP is disabled / on CPU)."""
    if not enabled or not _USE_CUDA or not ENABLE_AMP or _amp_autocast is None:
        import contextlib
        return contextlib.nullcontext()
    return _amp_autocast(**_AMP_DEVICE_KW)


# One global GradScaler per process; `enabled=False` makes it a no-op.
_scaler = _GradScaler(enabled=(_USE_CUDA and ENABLE_AMP))


# ---------------------------------------------------------------------------
# Weighted ensemble helpers
# ---------------------------------------------------------------------------

def _weighted_mean(values, weights):
    """
    Numerically-stable weighted mean. Returns 0.0 when the total weight is zero
    (i.e. all members are unsure) so the caller can still emit a 'FLAT' answer.
    """
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    s = float(w.sum())
    if s <= 0.0:
        return 0.0
    return float((v * w).sum() / s)


def _weighted_agreement(directions, weights):
    """
    Soft agreement in [0, 1].
      1.0  -> every member votes the same way
      0.0  -> a perfect 50/50 split (with equal weights)
    Computed as |sum(w * d) / sum(w)|, where d in {-1, +1}.
    """
    w = np.asarray(weights, dtype=np.float64)
    d = np.where(np.asarray(directions, dtype=bool), 1.0, -1.0)
    s = float(w.sum())
    if s <= 0.0:
        return 0.0
    return float(abs((w * d).sum()) / s)


def get_model_paths(symbol: str, suffix: str = ""):
    """Get model file paths. Optional suffix for ensemble models (e.g. '_ens0')."""
    symbol = symbol.upper()
    onnx_path = os.path.join(MODELS_DIR, f"{symbol}{suffix}.onnx")
    scaler_meta_path = os.path.join(MODELS_DIR, f"{symbol}{suffix}_scaler.json")
    return onnx_path, scaler_meta_path


def save_scaler_metadata(scaler, scaler_meta_path: str):
    # Support multi-feature scalers by saving lists
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    try:
        data_min_list = [float(x) for x in np.asarray(data_min).reshape(-1).tolist()]
        data_max_list = [float(x) for x in np.asarray(data_max).reshape(-1).tolist()]
    except Exception:
        # Ensure consistent type: always store as list (even single-element)
        data_min_list = [float(np.asarray(data_min).reshape(-1)[0])]
        data_max_list = [float(np.asarray(data_max).reshape(-1)[0])]

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
    """Normalize data using scaler metadata (data_min, data_max).

    Applies MinMax scaling: clips output to [0, 1] to match training.
    Works for both single-feature (1D/2D) and multi-feature arrays.
    """
    data_min = np.asarray(scaler_meta["data_min"], dtype=np.float64)
    data_max = np.asarray(scaler_meta["data_max"], dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    # Reshape mins/maxs for broadcasting (1, num_features)
    if data.ndim == 1:
        data_min = data_min.reshape(1, -1)
        data_max = data_max.reshape(1, -1)
    else:
        data_min = data_min.reshape(1, -1)
        data_max = data_max.reshape(1, -1)
    denom = data_max - data_min
    denom[denom == 0] = 1.0
    normalized = (data - data_min) / denom
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def inverse_with_metadata(data, scaler_meta):
    """Inverse-transform normalized predictions back to original scale.

    Reverses the MinMax scaling: x_orig = x_norm * (max - min) + min.
    Used to convert normalized price predictions back to actual prices.
    """
    data_min = np.asarray(scaler_meta["data_min"], dtype=np.float64)
    data_max = np.asarray(scaler_meta["data_max"], dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data_min = data_min.reshape(1, -1)
        data_max = data_max.reshape(1, -1)
    else:
        data_min = data_min.reshape(1, -1)
        data_max = data_max.reshape(1, -1)
    denom = data_max - data_min
    denom[denom == 0] = 1.0
    return (data * denom + data_min)


def _unwrap_for_export(model):
    """
    If `model` is wrapped by ``torch.compile`` (which exposes the original
    module on `_orig_mod`), return that. Otherwise return `model` as is.
    ONNX export must use the original, non-compiled module so the resulting
    graph does not contain Dynamo / Inductor wrappers.
    """
    return getattr(model, "_orig_mod", model)


def export_model_to_onnx(model: LSTMModel, onnx_path: str, num_features: int = 1, seq_length: int = None):
    model.eval()
    # Export from the underlying un-compiled module (torch.compile wrapper
    # exposes its target on `_orig_mod`) so the ONNX graph stays clean.
    export_target = _unwrap_for_export(model)
    # Ensure export happens on CPU to avoid device-specific artifacts
    cpu_model = export_target.to("cpu")
    actual_seq_len = seq_length if seq_length is not None else SEQ_LENGTH
    dummy_input = torch.randn(1, actual_seq_len, num_features, dtype=torch.float32).cpu()
    try:
        # Suppress the noisy PyTorch ONNX-exporter warning that fires for every
        # LSTM/GRU/RNN export via the legacy (dynamo=False) exporter. It is a
        # false positive when the model is already exported with batch_size=1
        # (which we do here) and inference always uses a single sequence.
        # See torch.onnx._internal.torchscript_exporter.symbolic_opset9._generic_rnn.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=re.escape(
                    "Exporting a model to ONNX with a batch_size other than 1, "
                    "with a variable length with "
                ),
                category=UserWarning,
            )
            torch.onnx.export(
                cpu_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=False,
                input_names=["inputs"],
                output_names=["prediction_return", "prediction_direction"],
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


def prepare_training_data(symbol: str, scaler=None, use_technical_features=True, seq_length=SEQ_LENGTH):
    """
    Fetch data, optionally add technical indicators, normalize, and create sequences.

    Returns:
        X:         input tensors
        y_returns: target return tensors
        y_dir:     target direction tensors
        scaler:    fitted scaler
        num_features: number of input features
        close_raw: raw close prices (for reference)
    """
    prices_df = fetch_stock_data(symbol)
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]

    if use_technical_features:
        # Augment with technical indicators (cached per symbol)
        features = get_cached_technical_features(symbol, prices_df)
    else:
        features = prices_df[feature_cols].values.astype(np.float32)

    # Fit scaler on the last NORMALIZE_DAYS rows and apply to full history
    fit_window = features[-NORMALIZE_DAYS:]
    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(fit_window)

    # Guard against division by zero
    denom = scaler.data_max_ - scaler.data_min_
    denom[denom == 0] = 1.0
    scaled_data = (features - scaler.data_min_) / denom
    prices_norm = np.clip(scaled_data, 0.0, 1.0).astype(np.float32)

    # Use raw close prices for computing returns
    close_raw = prices_df[["Close"]].values
    X, y_returns, y_direction = create_return_sequences(
        close_raw, prices_norm, seq_length=seq_length
    )

    if len(X) == 0:
        raise ValueError(f"Not enough valid return samples for {symbol}")

    return X, y_returns, y_direction, scaler, features.shape[1], close_raw


def _format_duration(seconds: float) -> str:
    """Format a number of seconds as e.g. '1h 02m 34s' or '12.3s'."""
    try:
        s = float(seconds)
    except Exception:
        return "?"
    if s < 0 or s != s:  # NaN / negative guard
        return "?"
    if s < 60:
        return f"{s:.1f}s"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


# ---------------------------------------------------------------------------
# On-the-fly augmentation (per-batch)
# ---------------------------------------------------------------------------
# Mirrors the old static `augment_sequences` semantics (3x effective dataset)
# but generates FRESH noise every batch instead of a static 3x tensor. Stacking
# [original | noisy | scaled] into one 3x batch is GPU-friendly: a single
# forward + backward pass per minibatch instead of three.
AUG_NOISE_STD = 0.01
AUG_SCALE_RANGE = (0.95, 1.05)


def _augment_batch_on_the_fly(xb, yb_ret, yb_dir):
    """Stack (original | noisy | scaled) into one 3x batch. Direction is
    preserved for the scaled variant (the target is scaled proportionally so
    its sign is unchanged)."""
    noise = torch.randn_like(xb) * AUG_NOISE_STD
    scale = float(np.random.uniform(*AUG_SCALE_RANGE))

    xb_noisy = xb + noise
    xb_scaled = xb * scale
    yb_ret_scaled = yb_ret * scale

    xb_aug = torch.cat([xb, xb_noisy, xb_scaled], dim=0)
    yb_ret_aug = torch.cat([yb_ret, yb_ret, yb_ret_scaled], dim=0)
    yb_dir_aug = torch.cat([yb_dir, yb_dir, yb_dir], dim=0)
    return xb_aug, yb_ret_aug, yb_dir_aug


def _maybe_compile(model: nn.Module) -> nn.Module:
    """
    Wrap `model` in ``torch.compile`` when supported. Falls back gracefully:
      - on older PyTorch without ``torch.compile``
      - on CPU-only systems (compile is CUDA-friendly and slow elsewhere)
      - when disabled via ENABLE_TORCH_COMPILE = False

    Compilation is for training only. ONNX export uses ``_unwrap_for_export``
    so the exported graph never contains the compiled wrapper.
    """
    if not ENABLE_TORCH_COMPILE:
        return model
    if not hasattr(torch, "compile"):
        return model
    if not _USE_CUDA:
        # Compiling on CPU is typically slower than eager mode for our shapes.
        return model
    try:
        return torch.compile(model)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("torch.compile failed (%s); using eager model.", e)
        return model


def train_model(
    symbol: str,
    X, y_returns, y_direction,
    num_features: int,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    seq_length=SEQ_LENGTH,
    epochs=EPOCHS_NEW,
    use_lr_scheduling=True,
    augment_data=True,
    progress_desc: str = None,
):
    """
    Core training loop with:
      * AMP (autocast + GradScaler); clean FP32 fallback on CPU
      * cuDNN benchmark enabled (set at import time)
      * torch.compile wrapper for the model (best-effort)
      * Huber + differentiable direction regression loss
      * Focal loss on the classification LOGITS (no sigmoid in the model)
      * Standard loss.backward() via the GradScaler; gradient surgery removed
      * Gradient clipping (after scaler.unscale_)
      * Cosine warm-restart scheduler (unchanged)
      * On-the-fly 3x batch augmentation (unchanged)
    """
    # DataLoader + on-the-fly augmentation
    X_t = torch.as_tensor(X, dtype=torch.float32).contiguous()
    y_ret_t_full = torch.as_tensor(y_returns, dtype=torch.float32).contiguous()
    y_dir_t_full = torch.as_tensor(y_direction, dtype=torch.float32).contiguous()

    dataset = TensorDataset(X_t, y_ret_t_full, y_dir_t_full)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE != "cpu"),
        drop_last=False,
    )

    model = LSTMModel(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(DEVICE)

    # Compile (best-effort) for training speed. ONNX export goes through
    # `_unwrap_for_export` so it sees the original module, not the wrapper.
    model = _maybe_compile(model)

    optimizer = Adam(model.parameters(), lr=LR)

    # Cosine annealing with warm restarts.
    scheduler = None
    if use_lr_scheduling:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=SCHEDULER_T_0,
            T_mult=SCHEDULER_T_MULT,
            eta_min=SCHEDULER_ETA_MIN,
        )

    # Build progress-bar description.
    bar_desc = progress_desc or f"{symbol}"

    model.train()
    start = time.time()
    pbar = tqdm(
        range(epochs),
        desc=bar_desc,
        ncols=110,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    last_dir_acc = 0.0
    for epoch in pbar:
        epoch_loss_sum = 0.0
        epoch_n = 0
        epoch_dir_correct = 0

        for xb, yb_ret, yb_dir in loader:
            # Async host->device copy (pin_memory makes this overlap with compute)
            xb = xb.to(DEVICE, non_blocking=True)
            yb_ret = yb_ret.to(DEVICE, non_blocking=True)
            yb_dir = yb_dir.to(DEVICE, non_blocking=True)

            # On-the-fly 3x augmentation (fresh noise every step).
            if augment_data:
                xb, yb_ret, yb_dir = _augment_batch_on_the_fly(
                    xb, yb_ret, yb_dir
                )

            optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast. On CPU / AMP disabled this is a
            # plain no-op context manager so the code path is identical.
            with _autocast_ctx(ENABLE_AMP):
                reg_pred, cls_logits = model(xb)
                # multi_task_loss now consumes raw logits (no sigmoid in model)
                # and applies the differentiable direction term inside its
                # regression loss. Gradient surgery has been removed.
                loss = multi_task_loss(
                    reg_pred, cls_logits, yb_ret, yb_dir,
                )

            # AMP-aware backward + step. The GradScaler is a no-op when AMP
            # is disabled, so this collapses to a plain `loss.backward()`.
            _scaler.scale(loss).backward()
            # Unscale so clip_grad_norm_ sees the real gradients.
            _scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )
            _scaler.step(optimizer)
            _scaler.update()

            bs = xb.size(0)
            epoch_loss_sum += loss.item() * bs
            epoch_n += bs
            with torch.no_grad():
                # The model now returns logits; apply sigmoid only for the
                # live direction-accuracy readout (not used for backprop).
                pred_dir_prob = torch.sigmoid(cls_logits.float())
                pred_dir_binary = (pred_dir_prob >= 0.5).float()
                epoch_dir_correct += (pred_dir_binary == yb_dir).sum().item()

        # Epoch-level metrics
        epoch_loss = epoch_loss_sum / max(1, epoch_n)
        epoch_acc = epoch_dir_correct / max(1, epoch_n)
        last_dir_acc = epoch_acc

        if scheduler is not None:
            scheduler.step(epoch)

        pbar.set_postfix(
            loss=f"{epoch_loss:.4f}",
            dir_acc=f"{epoch_acc:.2%}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    pbar.close()
    train_seconds = time.time() - start

    logger.info("Last dir_acc=%.3f", last_dir_acc)
    return model, train_seconds, last_dir_acc


def train_new_stock(symbol: str, epochs=EPOCHS_NEW, use_ensemble=False):
    """
    Train model(s) for a stock symbol with all directional accuracy improvements.

    Args:
        symbol: stock ticker
        epochs: number of training epochs
        use_ensemble: if True, train multiple models (ensemble)
    """
    if use_ensemble:
        return train_ensemble(symbol, epochs)

    # Single model training
    symbol = symbol.upper()

    X, y_returns, y_direction, scaler, num_features, _ = prepare_training_data(
        symbol, use_technical_features=True
    )

    model, train_seconds, final_dir_acc = train_model(
        symbol, X, y_returns, y_direction,
        num_features=num_features,
        epochs=epochs,
        use_lr_scheduling=True,
        augment_data=True,
    )

    # Save artifacts
    onnx_path, scaler_meta_path = get_model_paths(symbol)
    save_scaler_metadata(scaler, scaler_meta_path)
    try:
        export_model_to_onnx(model, onnx_path, num_features=num_features)
        logger.info("Saved model for %s to %s", symbol, onnx_path)
    except Exception as e:
        warnings.warn(f"ONNX export failed for {symbol}: {e}")


def retrain_stock_model(symbol: str):
    """
    Retrain a stock model from scratch (exports ONNX).
    Used by the API /update endpoint and the daily retrain job.
    """
    return train_new_stock(symbol.upper())


def train_ensemble(symbol: str, epochs=EPOCHS_NEW):
    """
    Train multiple models with varying architectures and sequence lengths.
    Each model is saved with a suffix like '_ens0', '_ens1', '_ens2'.

    Optimization: Fetch data and compute features ONCE, then slice sequences
    for each member instead of re-fetching/re-computing per member.
    """
    symbol = symbol.upper()
    logger.info("Training ensemble for %s with %d members", symbol, len(ENSEMBLE_CONFIGS))

    # Fetch data and compute features ONCE (data caching optimization)
    prices_df = fetch_stock_data(symbol)
    features = get_cached_technical_features(symbol, prices_df)
    close_raw = prices_df[["Close"]].values
    num_features = features.shape[1]

    # Hoist scaler fit and prices_norm ABOVE the loop
    # fit_window and the default-args MinMaxScaler are deterministic, so
    # `prices_norm` is bit-identical across members. Doing it once instead of
    # 3x removes the redundant NumPy normalize pass and the redundant Python
    # MinMaxScaler construction.
    fit_window = features[-NORMALIZE_DAYS:]
    from sklearn.preprocessing import MinMaxScaler
    scaler_member = MinMaxScaler()
    scaler_member.fit(fit_window)
    denom = scaler_member.data_max_ - scaler_member.data_min_
    denom[denom == 0] = 1.0
    scaled_data = (features - scaler_member.data_min_) / denom
    prices_norm = np.clip(scaled_data, 0.0, 1.0).astype(np.float32)

    # Per-seq_length cache for create_return_sequences
    # create_return_sequences is a Python for-loop over ~1200 rows. Memoize
    # by seq_length so each unique lookback is built once.
    seq_cache: dict = {}

    for idx, cfg in enumerate(ENSEMBLE_CONFIGS):
        suffix = f"_ens{idx}"
        seq_len = cfg["seq_length"]
        hidden_size = cfg["hidden_size"]
        num_layers = cfg["num_layers"]
        seed = cfg["seed"]

        logger.info(
            "Training ensemble member %d: seq_len=%d, hidden=%d, layers=%d, seed=%d",
            idx, seq_len, hidden_size, num_layers, seed,
        )

        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        if seq_len in seq_cache:
            X_mem, y_ret_mem, y_dir_mem = seq_cache[seq_len]
        else:
            X_mem, y_ret_mem, y_dir_mem = create_return_sequences(
                close_raw, prices_norm, seq_length=seq_len
            )
            seq_cache[seq_len] = (X_mem, y_ret_mem, y_dir_mem)

        model, train_seconds, final_dir_acc = train_model(
            symbol, X_mem, y_ret_mem, y_dir_mem,
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=seq_len,
            epochs=epochs,
            use_lr_scheduling=True,
            augment_data=True,
        )

        onnx_path, scaler_meta_path = get_model_paths(symbol, suffix=suffix)
        save_scaler_metadata(scaler_member, scaler_meta_path)

        # Export ensemble model with its own seq_length
        try:
            export_model_to_onnx(model, onnx_path, num_features=num_features, seq_length=seq_len)
            logger.info("Saved ensemble member %d to %s", idx, onnx_path)
        except Exception as e:
            warnings.warn(f"ONNX export failed for {symbol} ensemble member {idx}: {e}")

    logger.info("Ensemble training complete for %s", symbol)


def load_model(symbol: str, suffix: str = ""):
    onnx_path, scaler_meta_path = get_model_paths(symbol, suffix=suffix)
    scaler_meta = load_scaler_metadata(scaler_meta_path)
    onnx_session = create_onnx_session(onnx_path)
    if scaler_meta is None or onnx_session is None:
        return None, None
    return scaler_meta, onnx_session


def _try_load_predictions(symbol, prices_df=None):
    """
    Attempt to load ensemble models and/or single model for a symbol.
    Returns a list of prediction dicts (may be empty).

    If `prices_df` is supplied, it is reused across all ensemble members
    instead of refetching from yfinance for each one.
    """
    predictions = []

    # Try ensemble members first
    for idx in range(len(ENSEMBLE_CONFIGS)):
        suffix = f"_ens{idx}"
        scaler_meta, onnx_session = load_model(symbol, suffix=suffix)
        if scaler_meta is None or onnx_session is None:
            continue
        seq_len = ENSEMBLE_CONFIGS[idx]["seq_length"]
        try:
            pred = _predict_with_session(
                symbol, scaler_meta, onnx_session,
                seq_len=seq_len, prices_df=prices_df,
            )
            predictions.append(pred)
        except Exception as e:
            logger.warning("Ensemble member %d failed for %s: %s", idx, symbol, e)

    # Fallback to single model
    if not predictions:
        scaler_meta, onnx_session = load_model(symbol)
        if scaler_meta is not None and onnx_session is not None:
            pred = _predict_with_session(
                symbol, scaler_meta, onnx_session, prices_df=prices_df,
            )
            predictions.append(pred)

    return predictions


def predict_next_day(symbol: str):
    symbol = symbol.upper().strip()
    _predict_start = time.time()

    # ONE fetch per predict call -- reused across all ensemble members
    # (and across the first try / post-train re-try below).
    prices_df = fetch_stock_data(symbol)

    predictions = _try_load_predictions(symbol, prices_df=prices_df)
    mode = "inference"

    if not predictions:
        # Train from scratch then try again
        mode = "training"
        train_new_stock(symbol)
        # Training writes new rows to the CSV; refetch once to pick them up.
        prices_df = fetch_stock_data(symbol)
        predictions = _try_load_predictions(symbol, prices_df=prices_df)

    if not predictions:
        raise RuntimeError(f"Model artifacts are missing for {symbol}")

    # Robust weighted ensemble aggregation
    # Each member contributes:
    #   * a signed return  predicted_return  (regression head)
    #   * a direction bool predicted_direction (cls head, thresholded at 0.5)
    #   * a confidence     confidence = 2 * |p - 0.5|   in [0, 1]
    # Every aggregation step is weighted by confidence so a single loud but
    # unsure member can't dominate, and confident agreement raises both the
    # magnitude and the reported confidence. The sign is taken from a *soft*
    # weighted vote, not a hard majority, so a 2v1 split with a very sure
    # dissenter still nudges the answer toward the dissenter proportionally.
    returns     = np.asarray([p["predicted_return"]    for p in predictions], dtype=np.float64)
    directions  = np.asarray([bool(p["predicted_direction"]) for p in predictions])
    confidences = np.asarray([p["confidence"]          for p in predictions], dtype=np.float64)

    # Guard: a member with confidence 0 is effectively abstaining.
    # We don't drop it (keeps the math defined), but it carries no weight.
    total_weight = float(confidences.sum())
    if total_weight <= 0.0:
        # All members unsure -> emit a flat, zero-confidence prediction.
        base = predictions[0]
        _elapsed = time.time() - _predict_start
        return {
            "symbol": base["symbol"],
            "current_price": base["current_price"],
            "next_day_prediction": float(base["current_price"]),
            "predicted_return": 0.0,
            "direction": "FLAT",
            "confidence": 0.0,
            "ensemble_size": len(predictions),
            "agreement": 0.0,
            "soft_sign": 0.0,
            "mode": mode,
            "processing_time_seconds": round(_elapsed, 2),
        }

    # 1) Soft, weighted sign in [-1, +1].
    #    w_sign > 0  => net up
    #    w_sign < 0  => net down
    #    w_sign ~ 0  => ensemble is conflicted
    signs = np.where(directions, 1.0, -1.0)
    soft_sign = float((confidences * signs).sum() / total_weight)

    # 2) Soft agreement in [0, 1] (used both for magnitude and confidence).
    agreement = _weighted_agreement(directions, confidences)

    # 3) Weighted mean of |return| on the winning side, with the winning sign
    #    applied. Members on the losing side do not contribute to the magnitude
    #    (their loud magnitude is no longer diluted by zeroing it out -- it is
    #    simply not used, which is what "robust" means here).
    winning_sign = 1.0 if soft_sign >= 0.0 else -1.0
    if winning_sign > 0.0:
        winning_mask = directions
    else:
        winning_mask = ~directions

    if winning_mask.any():
        weighted_magnitude = _weighted_mean(
            np.abs(returns[winning_mask]),
            confidences[winning_mask],
        )
    else:
        # Defensive fallback: zero members on the winning side (only possible
        # for even-sized ensembles that are perfectly split).
        weighted_magnitude = _weighted_mean(np.abs(returns), confidences)

    final_return = weighted_magnitude * winning_sign

    # 4) Ensemble confidence = mean_conf * boosted_agreement.
    #    Old formula: mean_conf * agreement  (two low numbers → very low result).
    #    New formula: mean_conf * (0.3 + 0.7 * agreement)
    #    Full agreement (1.0) → multiplier 1.0 (no change).
    #    Partial disagreement (0.5) → multiplier 0.65 instead of 0.5.
    #    This prevents the ensemble from zeroing out confidence when individual
    #    members are reasonably sure but mildly disagree.
    mean_conf = _weighted_mean(confidences, confidences)
    boosted_agreement = 0.3 + 0.7 * agreement
    ensemble_confidence = max(0.0, float(mean_conf * boosted_agreement))

    base = predictions[0]
    current_price = base["current_price"]
    next_day_prediction = float(current_price * (1.0 + final_return))

    if final_return > 0.0:
        direction = "UP"
    elif final_return < 0.0:
        direction = "DOWN"
    else:
        direction = "FLAT"

    _elapsed = time.time() - _predict_start
    return {
        "symbol": symbol,
        "current_price": current_price,
        "next_day_prediction": next_day_prediction,
        "predicted_return": final_return,
        "direction": direction,
        "confidence": ensemble_confidence,
        "ensemble_size": len(predictions),
        # Additive new fields (backward-compatible with existing API consumers).
        "agreement": agreement,    # soft agreement in [0, 1]
        "soft_sign": soft_sign,    # continuous sign in [-1, +1]
        "mode": mode,
        "processing_time_seconds": round(_elapsed, 2),
    }


def _predict_with_session(symbol, scaler_meta, onnx_session, seq_length=SEQ_LENGTH, prices_df=None):
    """
    Run inference with an ONNX session.

    If `prices_df` is supplied, it is reused and no yfinance fetch is performed.
    Otherwise the function fetches data internally (legacy behavior).

    Returns dict with predicted_return, predicted_direction, current_price, confidence.
    """
    if prices_df is None:
        prices_df = fetch_stock_data(symbol)
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Build features with technical indicators to match training (cached per symbol)
    features = get_cached_technical_features(symbol, prices_df)
    num_features = features.shape[1]

    # Normalize using scaler metadata
    if isinstance(scaler_meta.get("data_min"), list) and \
       len(scaler_meta["data_min"]) == features.shape[1]:
        mins = np.asarray(scaler_meta["data_min"]).reshape(1, -1)
        maxs = np.asarray(scaler_meta["data_max"]).reshape(1, -1)
        denom = (maxs - mins)
        denom[denom == 0] = 1.0
        prices_norm_full = ((features - mins) / denom).astype(np.float32)
    else:
        # Fallback: use the select columns matching metadata
        mins = np.asarray(scaler_meta["data_min"]).reshape(1, -1)
        maxs = np.asarray(scaler_meta["data_max"]).reshape(1, -1)
        denom = (maxs - mins)
        denom[denom == 0] = 1.0
        base = prices_df[feature_cols].values.astype(np.float32)
        prices_norm_full = ((base - mins) / denom).astype(np.float32)
        num_features = base.shape[1]

    if len(prices_norm_full) < seq_length:
        raise ValueError(f"Not enough data for {symbol} (need {seq_length}, have {len(prices_norm_full)})")

    last_seq_np = np.expand_dims(
        prices_norm_full[-seq_length:].astype(np.float32), axis=0
    )
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: last_seq_np})

    current_price = float(prices_df["Close"].values[-1])

    # Output 0: predicted return (raw value)
    # Output 1: direction LOGIT (model now outputs raw logits, not sigmoid(p))
    pred_return = float(outputs[0][0][0])
    if len(outputs) > 1:
        z = float(outputs[1][0][0])
    else:
        z = 0.0

    # Direction decision: use RAW logit (temperature does not affect direction).
    pred_direction = z >= 0.0  # True = up, False = down

    # Confidence: apply temperature scaling BEFORE sigmoid to calibrate.
    # temperature < 1 stretches logits away from 0, producing higher confidence
    # without retraining. The direction threshold (z >= 0) is unchanged.
    cal_z = z / CONFIDENCE_TEMPERATURE if CONFIDENCE_TEMPERATURE > 0 else z
    cal_dir_prob = 1.0 / (1.0 + np.exp(-cal_z)) if cal_z >= 0 else \
                   np.exp(cal_z) / (1.0 + np.exp(cal_z))
    confidence = 2.0 * abs(cal_dir_prob - 0.5)  # [0, 1]

    return {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_return": pred_return,
        "predicted_direction": pred_direction,
        "confidence": confidence,
    }