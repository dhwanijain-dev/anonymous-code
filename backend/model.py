import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and inspect data
# -----------------------------
df = pd.read_csv("leak_dataset_timestep.csv", index_col=0)  # adjust path
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Identify pressure columns (all except metadata)
metadata_cols = ['scenario_id', 'label', 'leak_pipe', 'leak_node', 'leak_area']
pressure_cols = [col for col in df.columns if col not in metadata_cols]

# -----------------------------
# 2. Group by scenario_id and sort by time
# -----------------------------
sequences = []
labels = []
scenario_ids = []

for sid, group in df.groupby('scenario_id'):
    group = group.sort_index()                     # ensure time order (index = time)
    X = group[pressure_cols].values.astype(np.float32)
    y = group['label'].values.astype(np.float32)
    sequences.append(X)
    labels.append(y)
    scenario_ids.append(sid)

print(f"Number of scenarios: {len(sequences)}")
print(f"Sequence shape (first): {sequences[0].shape}")

# -----------------------------
# 3. Pad sequences to same length
#    (using numpy, later we'll mask padded steps)
# -----------------------------
from tensorflow.keras.preprocessing.sequence import pad_sequences  # optional, you can also use torch.nn.utils.rnn.pad_sequence

# Convert to list of tensors for easier handling later
sequences_torch = [torch.tensor(seq) for seq in sequences]
labels_torch = [torch.tensor(lab) for lab in labels]

# Pad sequences
X_padded = nn.utils.rnn.pad_sequence(sequences_torch, batch_first=True, padding_value=0.0)
y_padded = nn.utils.rnn.pad_sequence(labels_torch, batch_first=True, padding_value=-1.0)  # -1 will be ignored

print("Padded shape:", X_padded.shape)  # (n_scenarios, max_timesteps, n_features)

# -----------------------------
# 4. Train / validation / test split (by scenario)
# -----------------------------
n_scenarios = X_padded.shape[0]
indices = np.arange(n_scenarios)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)  # 0.25*0.8 = 0.2

X_train, y_train = X_padded[train_idx], y_padded[train_idx]
X_val, y_val = X_padded[val_idx], y_padded[val_idx]
X_test, y_test = X_padded[test_idx], y_padded[test_idx]

print(f"Train scenarios: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -----------------------------
# 5. Feature scaling (fit on training data only)
# -----------------------------
n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
X_train_flat = X_train.reshape(-1, n_features).numpy()
scaler = StandardScaler().fit(X_train_flat)

def scale_tensor(X):
    orig_shape = X.shape
    X_flat = X.reshape(-1, n_features).numpy()
    X_scaled = scaler.transform(X_flat)
    return torch.tensor(X_scaled.reshape(orig_shape), dtype=torch.float32)

X_train = scale_tensor(X_train)
X_val   = scale_tensor(X_val)
X_test  = scale_tensor(X_test)

# -----------------------------
# 6. Create PyTorch Dataset and DataLoader
# -----------------------------
class LeakDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = LeakDataset(X_train, y_train)
val_dataset   = LeakDataset(X_val, y_val)
test_dataset  = LeakDataset(X_test, y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 7. Define LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_dim)
        out = self.fc(lstm_out)              # (batch, seq_len, 1)
        out = self.sigmoid(out).squeeze(-1)  # (batch, seq_len)
        return out

# Hyperparameters
input_dim = n_features
hidden_dim = 64
num_layers = 2
dropout = 0.2

model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCELoss(reduction='none')  # we will mask padded steps
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 8. Training Loop with Masking
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)                     # (batch, seq_len)

        # Create mask: positions where y_batch != -1 are valid
        mask = (y_batch != -1).float()
        loss = criterion(outputs, y_batch)           # (batch, seq_len)
        loss = (loss * mask).sum() / mask.sum()      # average over valid timesteps

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            mask = (y_batch != -1).float()
            loss = criterion(outputs, y_batch)
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item()
    return total_loss / len(loader)

epochs = 60
train_losses, val_losses = [], []

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# -----------------------------
# 9. Evaluation on Test Set
# -----------------------------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)                         # (batch, seq_len)
        preds = (outputs > 0.5).float()
        # Flatten and collect only valid timesteps
        mask = (y_batch != -1)
        all_preds.append(preds[mask].cpu().numpy())
        all_labels.append(y_batch[mask].cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

torch.save(model.state_dict(),"model.pth")

print("\nClassification Report:")
print(all_labels,all_preds)
print(classification_report(all_labels, all_preds, target_names=['No leak', 'Leak']))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# -----------------------------
# 10. Plot Training History
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1,2,2)
# Accuracy over epochs (optional) – you can compute per epoch if desired
# Here we just show loss
plt.title('Training complete')
plt.tight_layout()
plt.show()