import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FOCAL_GAMMA, CLS_HIDDEN_DROPOUT,
    HUBER_DELTA, DIR_LOSS_WEIGHT, DIR_LOGIT_SCALE, CLS_LOSS_WEIGHT,
)


class LSTMModel(nn.Module):
    """
    Multi-task LSTM that predicts:
    1. Return magnitude (regression head)  -> raw return value
    2. Direction up/down (classification head) -> raw LOGIT

    The classification head outputs raw logits. Apply ``torch.sigmoid``
    at inference time only; the loss functions consume the logits directly
    so they are numerically stable and produce a clean gradient.
    """

    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_features_out=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=DROPOUT,
            batch_first=True
        )
        # Regression head: predict next-day return
        self.fc_reg = nn.Linear(hidden_size, 1)

        # Classification head (2-layer MLP with dropout) -> raw logits
        self.fc_cls_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.cls_dropout = nn.Dropout(CLS_HIDDEN_DROPOUT)
        self.fc_cls_2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            reg_out:    predicted return     [batch, 1]
            cls_logits: raw direction logit  [batch, 1]
                        (apply torch.sigmoid for probabilities)
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step only

        reg_out = self.fc_reg(out)                           # return magnitude

        # 2-layer classification head; produce logits (NO sigmoid here).
        cls_hidden = F.relu(self.fc_cls_1(out))
        cls_hidden = self.cls_dropout(cls_hidden)
        cls_logits = self.fc_cls_2(cls_hidden)               # raw logits

        return reg_out, cls_logits


# ────────────────────────────────────────────────────────────────────────────
# Regression loss: Huber + differentiable direction term
# ────────────────────────────────────────────────────────────────────────────

def regression_loss(reg_pred, target_returns, target_direction,
                    huber_delta=HUBER_DELTA,
                    dir_weight=DIR_LOSS_WEIGHT,
                    dir_logit_scale=DIR_LOGIT_SCALE):
    """
    Differentiable direction-aware regression loss.

        reg_loss = huber(reg_pred, target_returns)
                 + dir_weight * BCE-with-logits( reg_pred * dir_logit_scale,
                                                  target_direction )

    The first term is robust to outlier returns. The second term gives the
    model a *differentiable* signal to push `sign(reg_pred)` toward
    `sign(target_returns)`. Because we scale the regression logit by
    ``dir_logit_scale`` (100.0) before the BCE, even tiny predicted returns
    produce a non-trivial gradient when they point the wrong way.

    Args:
        reg_pred:         predicted return          [batch, 1]
        target_returns:   true return               [batch, 1]
        target_direction: true direction (1=up, 0=down) [batch, 1]
        huber_delta:      threshold for the Huber transition
        dir_weight:       weight of the direction term
        dir_logit_scale:  scaling applied to reg_pred before BCE
    Returns:
        scalar loss tensor
    """
    huber = F.huber_loss(reg_pred, target_returns, delta=huber_delta)

    # Differentiable direction term.
    #   target_direction in {0.0, 1.0}  ->  BCE-with-logits target
    #   input = reg_pred * scale       ->  acts like a sharp "is the sign right?"
    dir_term = F.binary_cross_entropy_with_logits(
        reg_pred * dir_logit_scale,
        target_direction,
    )
    return huber + dir_weight * dir_term


# ────────────────────────────────────────────────────────────────────────────
# Focal loss for binary classification, operating on LOGITS
# ────────────────────────────────────────────────────────────────────────────

def focal_binary_loss(logits, target, gamma=FOCAL_GAMMA):
    """
    Focal loss applied to raw logits (numerically stable via BCE-with-logits).

        FL(logits, t) = (1 - p_t)^gamma * BCEWithLogits(logits, t)

    where p_t = sigmoid(logits) if t==1 else 1 - sigmoid(logits).

    Args:
        logits: raw class logits [batch, 1]
        target: true labels      [batch, 1] (1.0 = up, 0.0 = down)
        gamma:  focusing parameter (0 -> plain BCE-with-logits)
    Returns:
        scalar loss tensor
    """
    # Per-sample BCE; reduction='none' so we can apply focal weighting.
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    # p_t: probability of the true class.
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)

    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def multi_task_loss(reg_pred, cls_logits, target_returns, target_direction,
                    dir_weight=DIR_LOSS_WEIGHT,
                    cls_weight=CLS_LOSS_WEIGHT,
                    use_focal=True,
                    huber_delta=HUBER_DELTA,
                    dir_logit_scale=DIR_LOGIT_SCALE):
    """
    Combined loss for the multi-task model.

        total = regression_loss(reg_pred, target_returns, target_direction)
              + cls_weight * ( focal or bce )( cls_logits, target_direction )

    Args:
        reg_pred:         predicted return     [batch, 1]
        cls_logits:       raw direction logits [batch, 1]
        target_returns:   true return          [batch, 1]
        target_direction: true direction       [batch, 1]
        dir_weight:       weight of the differentiable direction term
        cls_weight:       weight of the classification loss
        use_focal:        True -> focal, False -> plain BCE-with-logits
        huber_delta:      Huber threshold
        dir_logit_scale:  scale applied to reg_pred for the direction term
    """
    reg_loss = regression_loss(
        reg_pred, target_returns, target_direction,
        huber_delta=huber_delta,
        dir_weight=dir_weight,
        dir_logit_scale=dir_logit_scale,
    )

    if use_focal:
        cls_loss = focal_binary_loss(cls_logits, target_direction)
    else:
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target_direction)

    return reg_loss + cls_weight * cls_loss


# ────────────────────────────────────────────────────────────────────────────
# Deprecated: direction_aware_backprop (gradient surgery)
# ────────────────────────────────────────────────────────────────────────────
# The custom non-differentiable gradient amplifier has been removed because
# it (a) contributed almost no usable gradient and (b) made the training
# graph harder to follow / AMP-incompatible. The differentiable direction
# term inside ``regression_loss`` is the replacement.
#
# A no-op shim is kept so that any external import does not raise. It logs a
# one-time deprecation warning and just performs the standard backward pass.

def direction_aware_backprop(*args, **kwargs):  # pragma: no cover - shim
    """
    DEPRECATED. This function used to amplify gradients for wrong-direction
    samples after ``loss.backward()``. It has been replaced by the
    differentiable direction term in ``regression_loss`` and is now a
    no-op (it does nothing). Replace any call site with a plain
    ``loss.backward()`` (or the AMP-aware ``scaler.scale(loss).backward()``).
    """
    import warnings
    warnings.warn(
        "direction_aware_backprop is deprecated and has been removed. "
        "Use a standard loss.backward() (or scaler.scale(loss).backward() "
        "when using AMP) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None
