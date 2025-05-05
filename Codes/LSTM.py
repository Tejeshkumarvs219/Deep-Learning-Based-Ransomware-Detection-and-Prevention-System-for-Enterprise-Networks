import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Using device: {device}\n")

# Load the datasets
print(" Loading training dataset...")
train_data_dict = torch.load("D:/Masters_Project/Datasets/RanSAP/torch_train_dataset_fixed_correct.pt")
test_data_dict = torch.load("D:/Masters_Project/Datasets/RanSAP/torch_test_dataset_fixed_correct.pt")

# Convert to PyTorch tensors
X_train = torch.tensor(train_data_dict["data"], dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, 6)
y_train = torch.tensor(train_data_dict["labels"], dtype=torch.long).to(device)
X_test = torch.tensor(test_data_dict["features"], dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, 6)
y_test = torch.tensor(test_data_dict["labels"], dtype=torch.long).to(device)

print(f" Dataset Loaded: {X_train.shape}, Labels: {y_train.shape}\n")

# Define LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout after LSTM layer
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Initialize model
input_size = X_train.shape[2]
hidden_size = 128
num_layers = 2
num_classes = 26  # Number of ransomware families
dropout_rate = 0.3

model = LSTM(input_size, hidden_size, num_layers, num_classes, dropout_rate).to(device)
print(" Model initialized on", device)

# Training settings
batch_size = 1024
epochs = 20
learning_rate = 0.001
weight_decay = 1e-5  # L2 Regularization
patience = 3

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Early stopping tracking
best_val_loss = float('inf')
stopping_counter = 0

print(" Starting Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_acc = (correct / total) * 100
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader)
    print(f" Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        stopping_counter = 0
        torch.save(model.state_dict(), "D:/Masters_Project/Models/Optimized_LSTM.pth")
        print(" Model saved at D:/Masters_Project/Models/Optimized_LSTM.pth")
    else:
        stopping_counter += 1
        if stopping_counter >= patience:
            print(" Early Stopping Triggered!")
            break

print("\n Training Complete!")
