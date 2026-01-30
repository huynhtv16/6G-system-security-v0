import os
import re
import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc và xử lý dữ liệu từ server.log
log_lines = []
with open('result/server.log', 'r') as f:
    for line in f:
        if "Received Packet" in line and "|" in line:
            log_lines.append(line.strip())

# Trích xuất thông tin bằng regex
pattern = re.compile(
    r"Received Packet (?P<packet_id>\d+) \| "
    r"Source IP: (?P<source_ip>[\d.]+) \| "
    r"Destination IP: (?P<dest_ip>[\d.]+) \| "
    r"Source Port: (?P<source_port>\d+) \| "
    r"Destination Port: (?P<dest_port>\d+) \| "
    r"Protocol: (?P<protocol>\w+) \| "
    r"Flags: (?P<flags>[\w,]*) \| "
    r"TTL: (?P<ttl>\d+) \| "
    r"Volume: (?P<volume>\d+) \| "
    r"Spoofed: (?P<spoofed>[01]) \| "
    r"Attack Type: (?P<attack_type>\w+)"
)

data = []
for line in log_lines:
    match = pattern.search(line)
    if match:
        data.append(match.groupdict())

df = pd.DataFrame(data)

# Chuyển đổi kiểu dữ liệu
df['source_port'] = df['source_port'].astype(int)
df['dest_port'] = df['dest_port'].astype(int)
df['ttl'] = df['ttl'].astype(int)
df['volume'] = df['volume'].astype(int)
df['spoofed'] = df['spoofed'].astype(int)
df['flags'] = df['flags'].apply(lambda x: 1 if 'SYN' in x else 0)

# Mã hóa địa chỉ IP
df['source_ip'] = df['source_ip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
df['dest_ip'] = df['dest_ip'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# One-hot encode protocol
df = pd.get_dummies(df, columns=['protocol'], prefix='proto')

# Label encode attack_type
label_encoder = LabelEncoder()
df['attack_type'] = label_encoder.fit_transform(df['attack_type'])

# Chọn features và label
features = ['source_ip', 'dest_ip', 'source_port', 'dest_port', 'ttl', 'volume', 'spoofed', 'flags'] + [col for col in df.columns if 'proto_' in col]
X = df[features]
y = df['attack_type']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Định dạng lại cho mô hình CNN và RNN
n_features = X_train.shape[1]
X_train_cnn = X_train.reshape(-1, 1, 1, n_features)  # Reshape thành (samples, channels, height, width)
X_test_cnn = X_test.reshape(-1, 1, 1, n_features)
X_train_rnn = X_train.reshape(-1, 1, n_features)     # Reshape thành (samples, timesteps, features)
X_test_rnn = X_test.reshape(-1, 1, n_features)

# Chuyển sang Tensor
X_train_cnn_tensor = torch.FloatTensor(X_train_cnn)
X_test_cnn_tensor = torch.FloatTensor(X_test_cnn)
X_train_rnn_tensor = torch.FloatTensor(X_train_rnn)
X_test_rnn_tensor = torch.FloatTensor(X_test_rnn)
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Định nghĩa mô hình CNN
class CNNClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc1 = nn.Linear(16 * 1 * (n_features // 2), 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 1 * (n_features // 2))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Định nghĩa mô hình RNN
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Khởi tạo và huấn luyện mô hình
num_classes = len(label_encoder.classes_)
cnn_model = CNNClassifier(input_channels=1, num_classes=num_classes)
rnn_model = RNNClassifier(input_size=n_features, hidden_size=128, num_layers=2, num_classes=num_classes)

# Hàm huấn luyện
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.8f}")

# Tạo DataLoader
train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
train_dataset_rnn = TensorDataset(X_train_rnn_tensor, y_train_tensor)
train_loader_rnn = DataLoader(train_dataset_rnn, batch_size=64, shuffle=True)

# Huấn luyện mô hình
print("Training CNN Model...")
train_model(cnn_model, train_loader_cnn)

print("\nTraining RNN Model...")
train_model(rnn_model, train_loader_rnn)

# Đánh giá mô hình
def evaluate_model(model, X_test_tensor, y_test, model_name):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Sửa target_names sử dụng label encoder
    target_names = label_encoder.classes_  # Thay thế attack_mapping bằng này
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("Results", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # Sửa lỗi chính tả đường dẫn 'Resuilt' -> 'Results'
    with open(f"Results/{model_name}.txt", "w") as f:
        f.write(f"{model_name} Classifier:\nAccuracy: {accuracy:.2f}\nF1-Score: {f1:.2f}\n\n")
        f.write("Classification Report:\n" + report + "\n")
    
    # Vẽ confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_confusion_matrix.png")
    plt.close()
    
    return accuracy, f1

# Đánh giá và so sánh
cnn_accuracy, cnn_f1 = evaluate_model(cnn_model, X_test_cnn_tensor, y_test, "CNN")
rnn_accuracy, rnn_f1 = evaluate_model(rnn_model, X_test_rnn_tensor, y_test, "RNN")

# So sánh kết quả
print("\nComparison of Models:")
print(f"CNN - Accuracy: {cnn_accuracy:.2f}, F1-Score: {cnn_f1:.2f}")
print(f"RNN - Accuracy: {rnn_accuracy:.2f}, F1-Score: {rnn_f1:.2f}")

# Lưu mô hình
torch.save(cnn_model.state_dict(), "cnn_ddos_classifier.pth")
torch.save(rnn_model.state_dict(), "rnn_ddos_classifier.pth")
print("Models saved as 'cnn_ddos_classifier.pth' and 'rnn_ddos_classifier.pth'")