    
import argparse
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb

def topk_accuracy(probs, true_labels, k=3):
    # probs: (n_samples, n_classes) predicted probabilities
    # true_labels: (n_samples,) ints
    topk_preds = np.argsort(probs, axis=1)[:, ::-1][:, :k]  # top-k indices
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] in topk_preds[i]:
            correct += 1
    return correct / len(true_labels)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="gnn_rf_dataset.npz")
parser.add_argument("--out_dir", default="models")
parser.add_argument("--test_size", type=float, default=0.1)
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--rf_n_estimators", type=int, default=200)
parser.add_argument("--xgb_rounds", type=int, default=200)
parser.add_argument("--xgb_lr", type=float, default=0.1)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Load dataset
arr = np.load(args.dataset, allow_pickle=True)
X = arr["X"]   # shape (N, D)
y = arr["y"]   # shape (N,)
node_names = arr["node_names"]
edge_names = arr["edge_names"]

print("Loaded dataset:", X.shape, y.shape)
num_samples, feat_dim = X.shape
num_classes = int(np.max(y) + 1)
print("Num classes (pipes):", num_classes)

# Split: train / temp
idx_train, idx_temp, y_train, y_temp = train_test_split(np.arange(num_samples), y, test_size=(args.val_size + args.test_size), stratify=y, random_state=args.random_state)
# Split temp -> val / test
val_ratio = args.val_size / (args.val_size + args.test_size)
idx_val, idx_test, _, _ = train_test_split(idx_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=args.random_state)

X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

print("Splits: train", X_train.shape[0], "val", X_val.shape[0], "test", X_test.shape[0])

# Standardize? For tree models not necessary; but we center residuals optionally
# We'll scale features column-wise to zero mean for slight numeric stability
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-9
X_train_s = (X_train - mean) / std
X_val_s = (X_val - mean) / std
X_test_s = (X_test - mean) / std

# -------------------------
# RandomForest
# -------------------------
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=args.rf_n_estimators, n_jobs=-1, class_weight=None, random_state=args.random_state)
rf.fit(X_train_s, y_train)

probs_val_rf = rf.predict_proba(X_val_s)
val_top1_rf = accuracy_score(y_val, np.argmax(probs_val_rf, axis=1))
val_top3_rf = topk_accuracy(probs_val_rf, y_val, k=3)
print(f"RandomForest val Top-1: {val_top1_rf:.4f}, Top-3: {val_top3_rf:.4f}")

# Save RF
rf_path = os.path.join(args.out_dir, "rf_best.joblib")
joblib.dump({"model": rf, "mean": mean, "std": std, "node_names": node_names, "edge_names": edge_names}, rf_path)
print("Saved RandomForest to", rf_path)

# -------------------------
# XGBoost
# -------------------------
print("Training XGBoost...")
# xgboost expects DMatrix
dtrain = xgb.DMatrix(X_train_s, label=y_train)
dval = xgb.DMatrix(X_val_s, label=y_val)
param = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "eta": args.xgb_lr,
    "max_depth": 6,
    "verbosity": 1
}
evallist = [(dtrain, "train"), (dval, "val")]
bst = xgb.train(param, dtrain, num_boost_round=args.xgb_rounds, evals=evallist, early_stopping_rounds=20, verbose_eval=10)

# Evaluate
probs_val_xgb = bst.predict(dval)  # shape (n_val, num_classes)
val_top1_xgb = accuracy_score(y_val, np.argmax(probs_val_xgb, axis=1))
val_top3_xgb = topk_accuracy(probs_val_xgb, y_val, k=3)
print(f"XGBoost val Top-1: {val_top1_xgb:.4f}, Top-3: {val_top3_xgb:.4f}")

# Save XGBoost model
xgb_path = os.path.join(args.out_dir, "xgb_best.json")
bst.save_model(xgb_path)
# save scaler and names
joblib.dump({"mean": mean, "std": std, "node_names": node_names, "edge_names": edge_names}, os.path.join(args.out_dir, "xgb_meta.joblib"))
print("Saved XGBoost to", xgb_path)

# Final test evaluation with the better model (choose by val Top-1)
if val_top1_xgb >= val_top1_rf:
    print("XGBoost selected as best (val Top-1). Evaluating on test set...")
    dtest = xgb.DMatrix(X_test_s)
    probs_test = bst.predict(dtest)
    test_top1 = accuracy_score(y_test, np.argmax(probs_test, axis=1))
    test_top3 = topk_accuracy(probs_test, y_test, k=3)
    print(f"XGBoost test Top-1: {test_top1:.4f}, Top-3: {test_top3:.4f}")
else:
    print("RandomForest selected as best (val Top-1). Evaluating on test set...")
    probs_test_rf = rf.predict_proba(X_test_s)
    test_top1 = accuracy_score(y_test, np.argmax(probs_test_rf, axis=1))
    test_top3 = topk_accuracy(probs_test_rf, y_test, k=3)
    print(f"RF test Top-1: {test_top1:.4f}, Top-3: {test_top3:.4f}")

print("Done.")