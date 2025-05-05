import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys

# Load dataset
train_data = torch.load("D:/Masters_Project/Datasets/Ransap/torch_train_dataset_fixed_correct.pt")
X = train_data['data']
y = train_data['labels']

model_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'cnn'
assert model_type in ['cnn', 'lstm'], "Model type must be 'cnn' or 'lstm'"

# Define FCNN Model
class FCNN(nn.Module):
    def __init__(self, input_size=6, num_classes=26):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=26, dropout_rate=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, prec, rec, f1

# Cross-validation loop
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

accs, precs, recs, f1s = [], [], [], []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n Running Stratified 3-Fold Cross-Validation for {model_type.upper()} model...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f" Fold {fold+1}/3")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if model_type == 'cnn':
        model = FCNN().to(device)
        X_train_fold = X_train
        X_val_fold = X_val
    else:
        model = LSTM().to(device)
        X_train_fold = X_train.unsqueeze(1)
        X_val_fold = X_val.unsqueeze(1)

    train_dataset = TensorDataset(X_train_fold, y_train)
    val_dataset = TensorDataset(X_val_fold, y_val)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_model(model, train_loader, criterion, optimizer, device)
    acc, prec, rec, f1 = evaluate_model(model, val_loader, device)
    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
    print(f" Fold {fold+1} Results: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n")

print(" Final Average Metrics Across 3 Folds:")
print(f"   Accuracy : {np.mean(accs):.4f}")
print(f"   Precision: {np.mean(precs):.4f}")
print(f"   Recall   : {np.mean(recs):.4f}")
print(f"   F1-Score : {np.mean(f1s):.4f}")
