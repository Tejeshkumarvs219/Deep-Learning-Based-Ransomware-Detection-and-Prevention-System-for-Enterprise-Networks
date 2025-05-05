import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Hyperparameters
input_size = 6
num_classes = 26
batch_size = 1024
epochs = 20
learning_rate = 0.001

# Load Dataset
print(" Loading Dataset...")

train_data = torch.load("D:/Masters_Project/Datasets/Ransap/torch_train_dataset_fixed_correct.pt")
test_data = torch.load("D:/Masters_Project/Datasets/Ransap/torch_test_dataset_fixed_correct.pt")

X_train = train_data["data"]
y_train = train_data["labels"]
X_test = test_data["features"]
y_test = test_data["labels"]

print(" Dataset Loaded:", X_train.shape, y_train.shape)

# Dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# FCNN Model
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

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNN(input_size=input_size, num_classes=num_classes).to(device)
print(" Model initialized on", device)

# Optimizer & Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
print("\n Starting Training...")
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = running_loss / total

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f" Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "D:/Masters_Project/Models/FCNN_model.pth")
        print(" Model saved.")

print("\n Training Complete!")
