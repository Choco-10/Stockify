import torch
import torch.nn as nn
from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=DROPOUT,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # predict next day closing price

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # Take output of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
