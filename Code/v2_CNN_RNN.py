# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import random
import joblib

# Tạo thư mục nếu chưa tồn tại
os.makedirs("Resuilt", exist_ok=True)
os.makedirs("images", exist_ok=True)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

# Load và xử lý dữ liệu
csv_files = glob.glob('dataset/CSVs/01-12/*.csv')
data = pd.DataFrame()

def sample_csv(csv_file, sample_frac=1/30):
    # Đếm số dòng (trừ header)
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f) - 1
    n_sample = max(1, int(total_lines * sample_frac))
    skip = sorted(random.sample(range(1, total_lines + 1), total_lines - n_sample))
    df = pd.read_csv(csv_file, skiprows=skip, low_memory=False)
    return df

# Thay thế đoạn này trong vòng lặp:
for csv_file in csv_files:
    print(csv_file)
    df = sample_csv(csv_file, sample_frac=1/30)
    data = pd.concat([data, df], ignore_index=True)

data.columns = data.columns.str.strip()
data.dropna(inplace=True)

# Mapping các loại tấn công
attack_mapping = {
    'DrDoS_DNS': 0,
    'DrDoS_LDAP': 1,
    'DrDoS_MSSQL': 2,
    'DrDoS_NetBIOS': 3,
    'DrDoS_NTP': 4,
    'DrDoS_SNMP': 5,
    'DrDoS_SSDP': 6,
    'DrDoS_UDP': 7,
    'Syn': 8,
    'TFTP': 9,
    'UDPLag': 10
}

# Encode dữ liệu
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
unique_labels = np.unique(data['Label'])
print("Unique labels in dataset after encoding:", unique_labels)
encoded_to_original = {i: label_encoder.inverse_transform([i])[0] for i in unique_labels}
print("Encoded labels mapped to original:", encoded_to_original)

# Lọc dữ liệu
valid_labels = set(attack_mapping.values())
data = data[data['Label'].isin(valid_labels)]
print(f"Dataset size after filtering: {data.shape}")

# Tiếp tục xử lý dữ liệu
data['Source IP'] = data['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
data['Destination IP'] = data['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

data.replace([np.inf, -np.inf], [1e10, -1e10], inplace=True)
for column in data.columns:
    if not data[column].apply(is_float).all():
        data = data.drop(column, axis=1)

# Sau khi xử lý dữ liệu, trước khi chuẩn hóa:
feature_columns = list(data.drop(columns=['Label']).columns)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = data.drop(columns=['Label'])
y = data['Label']
X = scaler.fit_transform(X)

# Chia dữ liệu thành train, val, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Kiểm tra nhãn
num_classes = len(attack_mapping)
print("Max label in y_train:", y_train.max())
print("Max label in y_test:", y_test.max())
assert y_train.max() < num_classes, "y_train contains out-of-bound labels!"
assert y_test.max() < num_classes, "y_test contains out-of-bound labels!"

# Định dạng dữ liệu
n_features = X_train.shape[1]
side = int(np.ceil(np.sqrt(n_features)))
padding = side * side - n_features

# CNN data
X_train_cnn = np.pad(X_train, ((0, 0), (0, padding)), mode='constant').reshape(-1, 1, side, side)
X_val_cnn = np.pad(X_val, ((0, 0), (0, padding)), mode='constant').reshape(-1, 1, side, side)
X_test_cnn = np.pad(X_test, ((0, 0), (0, padding)), mode='constant').reshape(-1, 1, side, side)

# RNN data
X_train_rnn = X_train.reshape(-1, 1, n_features)
X_val_rnn = X_val.reshape(-1, 1, n_features)
X_test_rnn = X_test.reshape(-1, 1, n_features)

# Chuyển sang tensor
X_train_cnn_tensor = torch.FloatTensor(X_train_cnn).to(device)
X_val_cnn_tensor = torch.FloatTensor(X_val_cnn).to(device)
X_test_cnn_tensor = torch.FloatTensor(X_test_cnn).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
y_val_tensor = torch.LongTensor(y_val.values).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

X_train_rnn_tensor = torch.FloatTensor(X_train_rnn).to(device)
X_val_rnn_tensor = torch.FloatTensor(X_val_rnn).to(device)
X_test_rnn_tensor = torch.FloatTensor(X_test_rnn).to(device)

# DataLoader
train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
val_dataset_cnn = TensorDataset(X_val_cnn_tensor, y_val_tensor)
val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=64)

train_dataset_rnn = TensorDataset(X_train_rnn_tensor, y_train_tensor)
train_loader_rnn = DataLoader(train_dataset_rnn, batch_size=64, shuffle=True)
val_dataset_rnn = TensorDataset(X_val_rnn_tensor, y_val_tensor)
val_loader_rnn = DataLoader(val_dataset_rnn, batch_size=64)

# CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten_size = 32 * (side // 4) * (side // 4)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.001)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.001)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Khởi tạo mô hình
cnn_model = CNNClassifier(num_classes).to(device)
rnn_model = RNNClassifier(input_size=n_features, hidden_size=128, num_layers=3, num_classes=num_classes).to(device)

# Hàm huấn luyện
def train_model(model, train_loader, val_loader, model_name, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    with open(f"Resuilt/{model_name}_training_history.txt", "w") as f:
        f.write(f"Training History for {model_name}:\n")
        for ep, tl, ta, vl, va in zip(history['epoch'], history['train_loss'], history['train_acc'], history['val_loss'], history['val_acc']):
            f.write(f"Epoch {ep}/{num_epochs}, Train Loss: {tl:.4f}, Train Acc: {ta:.4f}, Val Loss: {vl:.4f}, Val Acc: {va:.4f}\n")
    
    torch.save(model.state_dict(), f"{MODEL_DIR}/{model_name.lower()}_ddos_classifier.pth")
    print(f"{model_name} model saved as '{model_name.lower()}_ddos_classifier.pth'")
    
    return history

# Huấn luyện
cnn_history = train_model(cnn_model, train_loader_cnn, val_loader_cnn, "CNN")
rnn_history = train_model(rnn_model, train_loader_rnn, val_loader_rnn, "RNN")

# Sau khi train xong, lưu scaler và feature_columns
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_cnn_rnn.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_columns_cnn_rnn.pkl')
joblib.dump(scaler, os.path.join(SCALER_PATH))
joblib.dump(feature_columns, os.path.join(FEATURE_COLS_PATH))

# Hàm đánh giá
def evaluate_model(model, X_test_tensor, y_test, model_name):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor.to(device))
        y_pred = torch.argmax(y_pred_tensor, dim=1).cpu().numpy()
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    target_names = [k for k, v in sorted(attack_mapping.items(), key=lambda x: x[1])]
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    result_text = f"{model_name} Classifier:\nAccuracy: {accuracy:.2f}\nF1-Score: {f1:.2f}\nInference Time: {inference_time:.4f} seconds\n\n"
    result_text += "Classification Report:\n" + report + "\n"
    with open(f"Resuilt/{model_name}.txt", "w") as f:
        f.write(result_text)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_confusion_matrix.png")
    plt.close()
    
    # Phân tích lỗi
    misclassified_idx = np.where(y_pred != y_test)[0]
    print(f"{model_name} - Number of misclassified samples: {len(misclassified_idx)}")
    with open(f"Resuilt/{model_name}_misclassifications.txt", "w") as f:
        f.write(f"Misclassified samples for {model_name} ({len(misclassified_idx)} total):\n")
        y_test_arr = np.array(y_test)  # Thêm dòng này trước vòng for

        for idx in misclassified_idx[:5]:
            true_label = list(attack_mapping.keys())[list(attack_mapping.values()).index(y_test_arr[idx])]
            pred_label = list(attack_mapping.keys())[list(attack_mapping.values()).index(y_pred[idx])]
            f.write(f"Sample {idx}: True={true_label}, Predicted={pred_label}\n")
    
    return accuracy, f1

# Đánh giá
cnn_accuracy, cnn_f1 = evaluate_model(cnn_model, X_test_cnn_tensor, y_test, "CNN")
rnn_accuracy, rnn_f1 = evaluate_model(rnn_model, X_test_rnn_tensor, y_test, "RNN")

print("\nComparison of Models:")
print(f"CNN - Accuracy: {cnn_accuracy:.2f}, F1-Score: {cnn_f1:.2f}")
print(f"RNN - Accuracy: {rnn_accuracy:.2f}, F1-Score: {rnn_f1:.2f}")
