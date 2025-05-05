import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Define CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, input_channels=1, num_features=6, num_classes=26):
        super(CNNClassifier, self).__init__()
        
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
            return sample_output.view(1, -1).size(1)  # Get flattened size

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

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Using device: {device}")

# Load test dataset
test_data_path = "D:/Masters_Project/Datasets/RanSAP/torch_test_dataset_fixed_correct.pt"
print("\n Loading test dataset...")
test_data_dict = torch.load(test_data_path)
X_test = test_data_dict["features"].float().unsqueeze(1).to(device)
Y_test = test_data_dict["labels"].to(device)
print(f" Test dataset loaded: {X_test.shape[0]} samples")

# Load trained model
model_path = "D:/Masters_Project/Models/Optimized_CNN.pth"
model = CNNClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"\n Model loaded from: {model_path}")

# Run inference
y_pred_probs = model(X_test)  # Forward pass
y_pred = torch.argmax(y_pred_probs, dim=1).cpu().numpy()
y_true = Y_test.cpu().numpy()

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n Evaluation Results:")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall   : {recall:.4f}")
print(f"   F1-Score : {f1:.4f}")

# Display Classification Report
print("\n Classification Report:")
print(classification_report(y_true, y_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 7))
sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
