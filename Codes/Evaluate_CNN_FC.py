import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
print(" Loading Test Dataset...")

test_data = torch.load("D:/Masters_Project/Datasets/Ransap/torch_test_dataset_fixed_correct.pt")
X_test = test_data["features"]
y_test = test_data["labels"]

print(" Test Dataset Loaded:", X_test.shape, y_test.shape)

# FCNN Model Definition
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

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNN(input_size=6, num_classes=26).to(device)
model.load_state_dict(torch.load("D:/Masters_Project/Models/FCNN_model.pth", map_location=device))
model.eval()
print(" Model Loaded.")

# Inference
print(" Running Inference...")

with torch.no_grad():
    outputs = model(X_test.to(device))
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu()

# Evaluation Metrics
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average="weighted", zero_division=0)
recall = recall_score(y_test, predicted, average="weighted", zero_division=0)
f1 = f1_score(y_test, predicted, average="weighted", zero_division=0)

print("\n Evaluation Results:")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall   : {recall:.4f}")
print(f"   F1-Score : {f1:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, predicted, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("D:/Masters_Project/Images/FCNN_Confusion_Matrix.png")
plt.show()
print(" Confusion Matrix Saved: D:/Masters_Project/Images/FCNN_Confusion_Matrix.png")
