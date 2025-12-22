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
HIDDEN_SIZE = 256     # hidden neurons per LSTM layer
NUM_LAYERS = 2        # stacked LSTM layers
LR = 0.001            # learning rate
DROPOUT = 0.2         # dropout to prevent overfitting
BATCH_SIZE = 16
EPOCHS_NEW = 100      # train new stock
EPOCHS_UPDATE = 5     # update existing model
NORMALIZE_DAYS = 500  # use last 500 days for normalization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
