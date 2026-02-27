import os
import joblib
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from prediction_data import build_dataset


# ---------------- CONFIG ----------------
DEFAULT_DATA_ROOT = "./BiWSData"   # BIWS scenario folder
MODEL_OUT = "model/detection.pkl"
# ----------------------------------------


def main() -> None:
    # Resolve data root
    data_root = os.environ.get("NET1_DATA_ROOT", DEFAULT_DATA_ROOT).strip()

    if not data_root:
        raise ValueError(
            "Data root not set. Either:\n"
            "1) Set NET1_DATA_ROOT env variable, or\n"
            "2) Edit DEFAULT_DATA_ROOT in the script."
        )

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

    print(f"Loading detection dataset from: {data_root}")

    # Each row = single timestamp snapshot (no temporal modeling)
    df, X, y = build_dataset(base_dir=data_root, horizon_steps=0)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("Class balance (0 = no leak, 1 = leak):")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    if len(unique) < 2 or 1 not in unique:
        raise ValueError(
            "No positive (leak) samples found. "
            "Check Labels.csv files for Label > 0."
        )

    # Train / test split
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

    print(f"Training samples → normal: {n_neg}, leak: {n_pos}")
    print(f"Using scale_pos_weight = {scale_pos_weight:.2f}")

    # XGBoost single-instance detector
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    print("Training XGBoost leak detection model...")
    model.fit(X_train, y_train)

    print("\n--- Detection Model Evaluation ---")
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nDetection model saved as '{MODEL_OUT}'")


if __name__ == "__main__":
    main()