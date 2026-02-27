"""
prediction_model.py
===================
Train the leak LOCALIZATION model (prediction.pkl).

Given a full-network pressure snapshot the model predicts:
  - Class 0  : no leak active right now
  - Class k>=1: leak at the junction named  node_id_map[k]

At inference (in main.py) the top-K class probabilities are returned so the
dashboard can show a ranked list of the most likely leak locations.

Outputs
-------
  model/prediction.pkl       — XGBoost multi-class classifier
  model/node_id_map.pkl      — list[str], index=class → junction node ID
                               (node_id_map[0] == 'no_leak')
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from node_prediction_data import build_localization_dataset


# ============================
# CONFIG
# ============================
DATA_ROOT       = "./BiWSData"
MODEL_OUT       = "model/prediction.pkl"
NODE_MAP_OUT    = "model/node_id_map.pkl"


def main() -> None:
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    print("===================================")
    print(" Training Leak Localization Model")
    print("===================================")
    print(f"Data root  : {DATA_ROOT}")
    print(f"Model out  : {MODEL_OUT}")
    print(f"Map out    : {NODE_MAP_OUT}\n")

    # ── Load multi-class dataset ───────────────────────────────────────
    print("Loading localization dataset ...")
    X, y, node_id_map = build_localization_dataset(DATA_ROOT)

    n_classes = len(node_id_map)   # 0 = no_leak, 1..K = leak nodes
    print(f"Feature matrix shape : {X.shape}")
    print(f"Label vector shape   : {y.shape}")
    print(f"Classes              : {n_classes}  (0=no_leak + {n_classes-1} unique leak nodes)")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass balance:")
    for cls, cnt in zip(unique, counts):
        label = node_id_map[cls] if cls < len(node_id_map) else f"class_{cls}"
        print(f"  class {cls:>4}  ({label:>12}) : {cnt}")

    if len(unique) < 2:
        raise ValueError(
            "Only one class found in the dataset.\n"
            "Re-generate BiWSData with leak scenarios present."
        )

    # ── Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\nTrain rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    # ── XGBoost multi-class ────────────────────────────────────────────
    model = xgb.XGBClassifier(
        objective        = "multi:softprob",
        num_class        = n_classes,
        n_estimators     = 200,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        eval_metric      = "mlogloss",
        n_jobs           = -1,
    )

    print("\nFitting XGBoost localization model ...")
    model.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────
    print("\n--- Evaluation on held-out data ---")
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Use node names as target_names for the report
    target_names = [f"cls{i}({node_id_map[i]})" for i in sorted(unique)]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model,       MODEL_OUT)
    joblib.dump(node_id_map, NODE_MAP_OUT)
    print(f"Localization model saved  → {MODEL_OUT}")
    print(f"Node ID map saved         → {NODE_MAP_OUT}")
    print(f"  (node_id_map[0]='no_leak', node_id_map[1..{n_classes-1}]=leak junction IDs)")


if __name__ == "__main__":
    main()