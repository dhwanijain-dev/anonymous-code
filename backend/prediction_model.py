import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

from prediction_data import build_dataset


# ============================
# DEFAULT CONFIG (EDIT HERE)
# ============================

DATA_ROOT    = "./BiWSData"         # Path to BiWSData root (Scenario-* folders)
HORIZON_STEPS = 0                   # 0 = current leak, >0 = future leak prediction
MODEL_OUT    = "model/prediction.pkl"   # Output model file


def main() -> None:
    # ----------------------------
    # Validate paths
    # ----------------------------
    if not DATA_ROOT or not DATA_ROOT.strip():
        raise ValueError("DATA_ROOT is empty. Please set it to the BiWSData directory.")

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Data root directory does not exist: {DATA_ROOT}")

    print("===================================")
    print(" Training Leak Prediction Model")
    print("===================================")
    print(f"Data root     : {DATA_ROOT}")
    print(f"Horizon steps : {HORIZON_STEPS}")
    print(f"Model output  : {MODEL_OUT}")

    # ----------------------------
    # Load dataset
    # ----------------------------
    print("\nLoading dataset...")
    df, X, y = build_dataset(
        base_dir=DATA_ROOT,
        horizon_steps=HORIZON_STEPS,
    )

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape : {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass balance (0 = no leak, 1 = leak):")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    if len(unique) < 2 or (1 not in unique) or counts[list(unique).index(1)] == 0:
        raise ValueError(
            "No positive (leak) samples found in the dataset.\n"
            "Check that Labels.csv files contain Label > 0."
        )

    # ----------------------------
    # Train / test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    if n_pos == 0:
        raise ValueError("No positive samples in training split.")

    scale_pos_weight = n_neg / n_pos
    print("\nTraining split:")
    print(f"  Normal samples: {n_neg}")
    print(f"  Leak samples  : {n_pos}")
    print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    # ----------------------------
    # Train XGBoost model
    # ----------------------------
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    print("\nFitting XGBoost model...")
    model.fit(X_train, y_train)

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\n--- Evaluation on held-out data ---")
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # ----------------------------
    # Save model
    # ----------------------------
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved successfully to: {MODEL_OUT}")


if __name__ == "__main__":
    main()