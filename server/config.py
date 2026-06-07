import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create folders if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# LSTM hyperparameters
SEQ_LENGTH = 60       # last 60 days as input
HIDDEN_SIZE = 256     # hidden neurons per LSTM layer (default for single-model path)
NUM_LAYERS = 2        # stacked LSTM layers
LR = 0.001            # learning rate
DROPOUT = 0.2         # dropout to prevent overfitting
# BATCH_SIZE raised from 16 -> 64 for better GPU utilization on RTX 4050 6GB.
# AMP + smaller ensemble members keep VRAM within budget.
BATCH_SIZE = 64
EPOCHS_NEW = 100      # train new stock
NORMALIZE_DAYS = 730  # use last 730 days for normalization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────────
# Directional accuracy improvements
# ────────────────────────────────────────────────────────────────────────────

# Multi-Task Learning weight for classification head
CLS_LOSS_WEIGHT = 0.5

# Focal Loss gamma (0 = standard BCE, higher focuses on hard examples)
FOCAL_GAMMA = 2.0

# Dropout for classification head hidden layer
CLS_HIDDEN_DROPOUT = 0.3

# Gradient clipping max norm
GRAD_CLIP_NORM = 1.0

# ────────────────────────────────────────────────────────────────────────────
# New losses & mixed-precision knobs
# ────────────────────────────────────────────────────────────────────────────

# Huber loss delta (in normalized-return space). 0.01 is conservative and
# makes the loss behave like MSE for small errors and like MAE for large ones.
HUBER_DELTA = 0.01

# Weight applied to the differentiable direction term inside the regression
# loss.  reg_loss = huber + DIR_LOSS_WEIGHT * direction_loss
DIR_LOSS_WEIGHT = 0.5

# Multiplier that scales `reg_pred` before feeding it to the BCE-with-logits
# direction loss. 100.0 effectively sharpens the sigmoid so small predicted
# returns still produce a useful gradient toward the correct sign.
DIR_LOGIT_SCALE = 100.0

# Post-hoc confidence calibration temperature.
# Applied as sigmoid(logit / temperature) instead of raw sigmoid(logit).
# temperature < 1 stretches logits away from 0 → higher confidence scores.
# 0.5 is a good default: a logit of 0.3 goes from 7.5% to ~29% confidence.
# This does NOT change the direction prediction (threshold stays at 0.5).
CONFIDENCE_TEMPERATURE = 0.5

# Enable CUDA automatic mixed precision (autocast + GradScaler).
# Falls back to plain FP32 on CPU or when CUDA is unavailable.
ENABLE_AMP = True

# Wrap the model in torch.compile when available (PyTorch >= 2.0).
# Compilation is applied for training only; ONNX export still uses the
# original (uncompiled) module so the exported graph stays clean.
ENABLE_TORCH_COMPILE = True

# Cosine annealing scheduler config
SCHEDULER_T_0 = 10       # initial restart period
SCHEDULER_T_MULT = 2     # period multiplier after each restart
SCHEDULER_ETA_MIN = 1e-5  # minimum learning rate

# ────────────────────────────────────────────────────────────────────────────
# Data download resilience
# ────────────────────────────────────────────────────────────────────────────
YF_RETRY_ATTEMPTS = 3    # max retries for yfinance downloads
YF_RETRY_BACKOFF = 2.0   # base seconds for exponential backoff (2 → 4 → 8)

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Feature engineering toggles
# ────────────────────────────────────────────────────────────────────────────

# Append SPY / QQQ market-regime return features (cached per stock).
MARKET_REGIME_ENABLED = True
MARKET_REGIME_SYMBOLS = ["SPY", "QQQ"]

# ────────────────────────────────────────────────────────────────────────────
# Ensemble training configs (3 models with varying lookback/depth)
# All members now use hidden_size=128 to reduce VRAM and overfitting while
# keeping the seq_length diversity. num_layers stays at 2.
# ────────────────────────────────────────────────────────────────────────────
ENSEMBLE_CONFIGS = [
    {"seq_length": 30,  "hidden_size": 128, "num_layers": 2, "seed": 42},
    {"seq_length": 60,  "hidden_size": 128, "num_layers": 2, "seed": 123},
    {"seq_length": 90,  "hidden_size": 128, "num_layers": 2, "seed": 456},
]
