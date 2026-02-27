"""
lstm_model.py
=============
Train the LSTM time-to-burst regression model  (model/lstm_burst.pt).

Architecture
------------
  Input  : (batch, SEQ_LEN=10, N_FEATURES=10)
            Sliding window of relative pressure features for ONE node.
  LSTM   : 2 layers, hidden_size=64, dropout=0.3.
  Head   : Linear(64 → 32) → ReLU → Dropout → Linear(32 → 1)
            NO final activation — raw regression output.
  Output : predicted time-to-burst in timesteps (float ≥ 0).

            0            → node is currently bursting
            1..HORIZON-1 → burst predicted in N timesteps
            ≥ HORIZON    → no near-term burst risk

Interpretation at inference
----------------------------
Run on all N_NODES simultaneously → shape (N_NODES,).
Sort ASCENDING → nodes with the SMALLEST predicted time are most urgent.

Loss
----
Huber loss (delta=1.0) — behaves like L2 near zero (sensitive to small
errors when burst is imminent) and L1 for large values (robust to the
many HORIZON-labelled non-bursting samples).

Saved artefacts
---------------
  model/lstm_burst.pt     — PyTorch state_dict (weights only)
  model/lstm_config.pkl   — dict with all architecture & data hyper-params
                            used to rebuild the model at inference time.
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from lstm_data import build_lstm_dataset, SEQ_LEN, N_FEATURES, HORIZON

# ============================
# CONFIG
# ============================
DATA_ROOT   = "./BiWSData"
MODEL_OUT   = "model/lstm_burst.pt"
CONFIG_OUT  = "model/lstm_config.pkl"

HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 512
EPOCHS      = 40
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# MODEL DEFINITION
# ============================
class LeakLSTM(nn.Module):
    """
    LSTM that predicts time-to-burst (regression, no sigmoid).
    Output is a single unbounded float per sample.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        # Regression head — NO sigmoid
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # Raw output; clamp at inference to [0, HORIZON]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # (batch,)


# ============================
# TRAINING
# ============================
def train() -> None:
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    print("=" * 50)
    print(" LSTM Time-to-Burst Regression Trainer")
    print("=" * 50)
    print(f"Device       : {DEVICE}")
    print(f"Seq len      : {SEQ_LEN}  |  Features : {N_FEATURES}")
    print(f"Horizon      : {HORIZON} timesteps  ({HORIZON} minutes)")
    print(f"Hidden size  : {HIDDEN_SIZE}  |  Layers : {NUM_LAYERS}")
    print(f"Loss         : Huber (delta=1.0)")

    # ── Build dataset ──────────────────────────────────────────────────
    print("\nBuilding regression dataset ...")
    X, y = build_lstm_dataset(DATA_ROOT)
    print(f"X shape  : {X.shape}  (samples × seq_len × features)")
    print(f"y range  : [{y.min():.1f}, {y.max():.1f}]  "
          f"mean={y.mean():.1f}  median={np.median(y):.1f}")
    print(f"  y=0 (bursting now)   : {(y == 0).sum()}")
    print(f"  y in (0, {HORIZON})   : {((y > 0) & (y < HORIZON)).sum()}")
    print(f"  y={HORIZON} (no risk) : {(y == HORIZON).sum()}")

    # ── Train / val split ──────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"\nTrain : {len(X_tr)} samples  |  Val : {len(X_val)} samples")

    # ── Weighted sampler — oversample imminent-burst rows ───────────────
    # Rows with small y (y < HORIZON/2) are up-weighted so the model
    # pays more attention to the critical near-burst region.
    urgency_weight = np.where(y_tr < HORIZON / 2, 3.0, 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(
        torch.tensor(urgency_weight),
        num_samples=len(urgency_weight),
        replacement=True,
    )

    train_ds = TensorDataset(
        torch.tensor(X_tr,  dtype=torch.float32),
        torch.tensor(y_tr,  dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model, loss, optimiser ─────────────────────────────────────────
    model     = LeakLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)          # robust regression loss
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=4, factor=0.5, verbose=True,
    )

    print(f"\nTraining for {EPOCHS} epochs ...")
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_ds)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss, all_pred, all_true = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred   = model(Xb)
                val_loss += criterion(pred, yb).item() * len(Xb)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(yb.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        all_pred = np.clip(np.array(all_pred), 0, HORIZON)
        all_true = np.array(all_true)
        mae      = np.abs(all_pred - all_true).mean()

        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"MAE={mae:.2f} ts")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Final evaluation ──────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            pred = model(Xb.to(DEVICE)).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(yb.numpy())
    all_pred = np.clip(np.array(all_pred), 0, HORIZON)
    all_true = np.array(all_true)

    print("\n--- Best model — validation set ---")
    print(f"  MAE          : {np.abs(all_pred - all_true).mean():.2f} timesteps")
    print(f"  RMSE         : {np.sqrt(((all_pred - all_true)**2).mean()):.2f} timesteps")

    # Accuracy within ±2 timesteps (for near-burst samples)
    near_mask = all_true < HORIZON / 2
    if near_mask.sum() > 0:
        acc2 = (np.abs(all_pred[near_mask] - all_true[near_mask]) <= 2).mean()
        print(f"  ±2-ts acc (urgent samples): {100*acc2:.1f}%")

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    config = {
        "input_size":  N_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "num_layers":  NUM_LAYERS,
        "dropout":     DROPOUT,
        "seq_len":     SEQ_LEN,
        "horizon":     HORIZON,
        "task":        "regression",   # marks this as time-to-burst, not classifier
    }
    joblib.dump(config, CONFIG_OUT)
    print(f"\nModel saved  → {MODEL_OUT}")
    print(f"Config saved → {CONFIG_OUT}")
    print(f"  Output: single float per node = predicted timesteps until burst")
    print(f"  Sort ASCENDING at inference — lowest value = most urgent node")


if __name__ == "__main__":
    train()
