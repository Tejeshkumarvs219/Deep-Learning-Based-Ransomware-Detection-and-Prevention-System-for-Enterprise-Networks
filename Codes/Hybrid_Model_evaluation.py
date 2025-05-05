import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# CNN-LSTM Hybrid Model
class CNNLSTM(nn.Module):
    def __init__(self, input_size=6, cnn_channels=32, lstm_hidden_size=64, lstm_layers=1, num_classes=26):
        super(CNNLSTM, self).__init__()
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

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:/Masters_Project/Models/CNN_LSTM_model.pth"
test_data_path = "D:/Masters_Project/Datasets/RanSAP/torch_test_dataset_fixed_correct.pt"

# Load Test Data
print(" Loading test dataset...")
test_data_dict = torch.load(test_data_path)
X_test = test_data_dict.get("features", test_data_dict.get("data"))
y_test = test_data_dict["labels"]

X_test = X_test.clone().detach().unsqueeze(1).to(device)  # Shape: (N, 1, 6)
y_test = y_test.clone().detach().to(device)
print(f" Test dataset loaded: {X_test.shape}")

# Load Model
model = CNNLSTM().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"\n Model loaded from: {model_path}")

# Inference
print("\n Running inference on test data...")
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# Evaluation Metrics
accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
precision = precision_score(y_test.cpu(), predicted.cpu(), average='weighted', zero_division=0)
recall = recall_score(y_test.cpu(), predicted.cpu(), average='weighted', zero_division=0)
f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted', zero_division=0)

print("\n Evaluation Results:")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall   : {recall:.4f}")
print(f"   F1-Score : {f1:.4f}")

# Detailed Classification Report
print("\n Classification Report:")
print(classification_report(y_test.cpu(), predicted.cpu(), digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test.cpu(), predicted.cpu())
plt.figure(figsize=(14, 8))
heatmap = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title("CNN-LSTM Hybrid - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar(heatmap.collections[0])
plt.tight_layout()
plt.savefig("CNN_LSTM_Confusion_Matrix.png")
plt.show()
