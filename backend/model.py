import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Data
print("Loading dataset...")
df = pd.read_csv("simulation_data.csv")

# 2. Preprocess the Data
# We drop 'timestamp' and 'node_id' because we want the model to learn 
# the physics of a leak, not memorize specific times or node names.
X = df.drop(columns=['timestamp', 'node_id', 'is_leaking'])
y = df['is_leaking']

# 3. Handle the Imbalanced Data (Crucial Step)
# We need to tell XGBoost to care more about the rare '1's (leaks).
# We do this by calculating the ratio of normal rows to leak rows.
num_negative_samples = (y == 0).sum()
num_positive_samples = (y == 1).sum()

if num_positive_samples == 0:
    raise ValueError("No leaks found in your dataset yet! Let the simulation run longer.")

scale_weight = num_negative_samples / num_positive_samples
print(f"Data balance - Normal: {num_negative_samples}, Leaks: {num_positive_samples}")
print(f"Applying scale_pos_weight of: {scale_weight:.2f}")

# 4. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Initialize the XGBoost Model
# We pass the scale_weight here so the model penalizes itself heavily if it misses a leak
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric='logloss'
)

# 6. Train the Model
print("Training the XGBoost model...")
model.fit(X_train, y_train)

# 7. Evaluate the Model
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# A confusion matrix shows True Positives, False Positives, etc.
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# The classification report shows Precision, Recall, and F1-Score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the Model
# We save the trained model as a file so your FastAPI app can use it later
model_filename = "xgboost_leak_model.pkl"
joblib.dump(model, model_filename)
print(f"\nModel saved successfully as '{model_filename}'")