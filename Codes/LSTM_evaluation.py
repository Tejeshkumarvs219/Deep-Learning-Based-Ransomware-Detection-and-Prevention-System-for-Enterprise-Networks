import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}\n")

# Load test data
print(" Loading test dataset...")
test_data_dict = torch.load("D:/Masters_Project/Datasets/RanSAP/torch_test_dataset_fixed_correct.pt")
X_test = torch.tensor(test_data_dict["features"], dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(test_data_dict["labels"], dtype=torch.long).to(device)
print(f" Test dataset loaded: {X_test.shape}")

# Load the trained model
model_path = "D:/Masters_Project/Models/Optimized_LSTM.pth"
input_size = 6
hidden_size = 128
num_layers = 2
num_classes = 26

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"\n Model loaded from: {model_path}")

# Run inference
print("\n Running inference on test data...")
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

# Convert tensors to CPU numpy arrays
y_true = y_test.cpu().numpy()
y_pred = predicted.cpu().numpy()

# Evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted")

print("\n Evaluation Results:")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall   : {recall:.4f}")
print(f"   F1-Score : {f1:.4f}")

# Classification report
print("\n Classification Report:")
print(classification_report(y_true, y_pred))

conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 8))
heatmap = sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=range(26), yticklabels=range(26))
plt.title("LSTM Model - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.colorbar(heatmap.collections[0])
plt.tight_layout()
plt.show()
