"""
lstm_model.py
=============
Train the LSTM burst predictor (model/lstm_burst.pt).

Architecture
------------
  Input  : (batch, SEQ_LEN=10, N_FEATURES=10) — sliding window of relative
            pressure features for ONE node.
  LSTM   : 2 layers, hidden_size=64, dropout=0.3.
  Head   : Linear(64 → 32) → ReLU → Dropout → Linear(32 → 1) → Sigmoid.
  Output : P(this node will burst within the next HORIZON=5 timesteps).

At inference (main.py) we run this for ALL 2,859 BIWS junctions in one
batched forward pass and return a ranked list of the most at-risk nodes.

Saved artefacts
---------------
  model/lstm_burst.pt     — trained PyTorch weights (state_dict)
  model/lstm_config.pkl   — {seq_len, n_features, hidden_size, num_layers}
                            needed to rebuild the model at inference time.
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from lstm_data import build_lstm_dataset, SEQ_LEN, N_FEATURES, HORIZON

# ============================
# CONFIG
# ============================
DATA_ROOT    = "./BiWSData"
MODEL_OUT    = "model/lstm_burst.pt"
CONFIG_OUT   = "model/lstm_config.pkl"

HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 512
EPOCHS       = 30
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# MODEL DEFINITION
# ============================
class LeakLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
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

    print("=" * 43)
    print(" Training LSTM Burst Predictor")
    print("=" * 43)
    print(f"Device       : {DEVICE}")
    print(f"Seq len      : {SEQ_LEN}  |  Features: {N_FEATURES}")
    print(f"Horizon      : {HORIZON} timesteps ahead")
    print(f"Hidden size  : {HIDDEN_SIZE}  |  Layers: {NUM_LAYERS}")

    # ── Load dataset ──────────────────────────────────────────────────
    print("\nBuilding LSTM dataset (sliding window sequences)...")
    X, y = build_lstm_dataset(DATA_ROOT)
    u, c = np.unique(y, return_counts=True)
    print(f"X shape      : {X.shape}  (samples × seq_len × features)")
    print(f"y shape      : {y.shape}")
    print("Class balance:")
    for cls, cnt in zip(u, c):
        name = "will_burst" if cls == 1 else "no_burst"
        print(f"  {int(cls)} ({name}): {cnt} ({100*cnt/len(y):.1f}%)")

    # ── Train / val split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # ── Weighted sampler to handle class imbalance ────────────────────
    n_neg  = (y_train == 0).sum()
    n_pos  = (y_train == 1).sum()
    w_pos  = n_neg / (n_pos + 1e-6)
    w_neg  = 1.0
    sample_weights = torch.tensor(
        [w_pos if yi == 1 else w_neg for yi in y_train], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model, loss, optimiser ────────────────────────────────────────
    model = LeakLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=3, factor=0.5, verbose=True,
    )

    print(f"\nTraining for {EPOCHS} epochs...")
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            preds = model(Xb)
            loss  = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                probs = model(Xb)
                val_loss += criterion(probs, yb).item() * len(Xb)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")

        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  AUC={auc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Evaluate best model ───────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    y_prob, y_true = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            p = model(Xb.to(DEVICE)).cpu().numpy()
            y_prob.extend(p)
            y_true.extend(yb.numpy())

    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    print("\n--- Best model evaluation (validation set) ---")
    print(classification_report(y_true, y_pred,
                                 target_names=["no_burst", "will_burst"], digits=3))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    except Exception:
        pass

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
    }
    joblib.dump(config, CONFIG_OUT)
    print(f"\nLSTM model saved   → {MODEL_OUT}")
    print(f"Config saved       → {CONFIG_OUT}")
    print(f"  Horizon = {HORIZON} timesteps  "
          f"= {HORIZON} minutes ahead (with 60s timestep)")


if __name__ == "__main__":
    train()
