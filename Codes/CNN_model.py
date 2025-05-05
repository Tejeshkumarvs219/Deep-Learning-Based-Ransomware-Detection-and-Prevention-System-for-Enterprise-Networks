import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Using device: {device}")

# Load datasets
train_data_path = "D:/Masters_Project/Datasets/RanSAP/torch_train_dataset_fixed_correct.pt"
test_data_path = "D:/Masters_Project/Datasets/RanSAP/torch_test_dataset_fixed_correct.pt"

print("\n Loading training dataset...")
train_data_dict = torch.load(train_data_path)
X_train = train_data_dict["data"].float().unsqueeze(1).to(device)  # Shape: (N, 1, 6)
y_train = train_data_dict["labels"].long().to(device)

print(" Loading test dataset...")
test_data_dict = torch.load(test_data_path)
X_test = test_data_dict["features"].float().unsqueeze(1).to(device)
y_test = test_data_dict["labels"].long().to(device)

print(f" Dataset Loaded: {X_train.shape}, Labels: {y_train.shape}")

# Create PyTorch dataset
train_dataset = data.TensorDataset(X_train, y_train)
test_dataset = data.TensorDataset(X_test, y_test)

# Define DataLoaders
batch_size = 8192
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# CNN Model
class CNN(nn.Module):
    def __init__(self, input_channels=1, num_features=6, num_classes=26):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        self.dropout = nn.Dropout(0.4)

        self.flatten_size = self._get_flatten_size(num_features)

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _get_flatten_size(self, num_features):
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, num_features)  # Simulated input
            sample_output = self._forward_conv_layers(sample_input)
            return sample_output.view(1, -1).size(1)

    def _forward_conv_layers(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model
model = CNN().to(device)
print(f"\n Model initialized on {device}")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Training Loop
num_epochs = 15
best_val_loss = float("inf")
model_save_path = "D:/Masters_Project/Models/Optimized_CNN.pth"

print("\n Starting Training...\n")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    
    # Validation Step
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total
    
    print(f" Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4%}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f" Model saved at {model_save_path}")

end_time = time.time()
print(f"\n Training Complete in {(end_time - start_time)/60:.2f} minutes!")

