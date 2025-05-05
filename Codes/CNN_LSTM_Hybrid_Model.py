import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# CNN-LSTM Hybrid Model Definition
class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_size=6, cnn_channels=32, lstm_hidden_size=64, lstm_layers=1, num_classes=26):
        super(CNNLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Paths
train_path = "D:/Masters_Project/Datasets/RanSAP/torch_train_dataset_fixed_correct.pt"
test_path = "D:/Masters_Project/Datasets/RanSAP/torch_test_dataset_fixed_correct.pt"
model_save_path = "D:/Masters_Project/Models/CNN_LSTM_model.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Using device: {device}")

# Load Dataset
print("\n Loading training dataset...")
train_data_dict = torch.load(train_path)
X_train = train_data_dict["data"].unsqueeze(1).float().to(device)
y_train = train_data_dict["labels"].long().to(device)

print(" Loading test dataset...")
test_data_dict = torch.load(test_path)
X_test = test_data_dict["features"].unsqueeze(1).float().to(device)
y_test = test_data_dict["labels"].long().to(device)

print(f" Dataset Loaded: {X_train.shape}, Labels: {y_train.shape}")

# DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize Model
model = CNNLSTMClassifier().to(device)
print("\n Model initialized on", device)

# Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("\n Starting Training...")
epochs = 20
best_val_acc = 0.0
early_stop_counter = 0
patience = 3
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100

    # Validation Accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_labels in test_loader:
            val_outputs = model(val_inputs)
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)
    val_acc = val_correct / val_total * 100

    print(f" Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f" Model saved at {model_save_path}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(" Early Stopping Triggered!")
            break

end_time = time.time()
print(f"\n Training Complete in {(end_time - start_time)/60:.2f} minutes!")
