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

# Kiểm tra xem có GPU khả dụng không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Sử dụng CPU vì không có GPU khả dụng.")

# Load và xử lý dữ liệu
csv_files = glob.glob('dataset/CSVs/01-12/*.csv')
data = pd.DataFrame()

for csv_file in csv_files:
    print(csv_file)
    df = pd.read_csv(csv_file)
    df = df.sample(frac=1/300, random_state=42)
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

# Kiểm tra các nhãn thực tế trong dữ liệu
unique_labels = np.unique(data['Label'])
print("Unique labels in dataset after encoding:", unique_labels)

# Reverse mapping để kiểm tra nhãn gốc
reverse_mapping = {v: k for k, v in attack_mapping.items()}
encoded_to_original = {i: label_encoder.inverse_transform([i])[0] for i in unique_labels}
print("Encoded labels mapped to original:", encoded_to_original)

# Lọc dữ liệu chỉ giữ lại các nhãn trong attack_mapping
valid_labels = set(attack_mapping.values())
data = data[data['Label'].isin(valid_labels)]
print(f"Dataset size after filtering invalid labels: {data.shape}")

# Tiếp tục xử lý dữ liệu
data['Source IP'] = data['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
data['Destination IP'] = data['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

# Xử lý dữ liệu số
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

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
excluded_columns = ['Label']
X = data.drop(columns=excluded_columns)
y = data['Label']
X = scaler.fit_transform(X)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kiểm tra nhãn tối đa để đảm bảo không vượt quá num_classes
num_classes = len(attack_mapping)
print("Max label in y_train:", y_train.max())
print("Max label in y_test:", y_test.max())
assert y_train.max() < num_classes, "y_train contains out-of-bound labels!"
assert y_test.max() < num_classes, "y_test contains out-of-bound labels!"

# Định dạng lại dữ liệu cho CNN (giả định thành ma trận vuông)
n_features = X_train.shape[1]
side = int(np.ceil(np.sqrt(n_features)))
padding = side * side - n_features
X_train_padded = np.pad(X_train, ((0, 0), (0, padding)), mode='constant')
X_test_padded = np.pad(X_test, ((0, 0), (0, padding)), mode='constant')
X_train_cnn = X_train_padded.reshape(-1, 1, side, side)  # Thêm channel dimension
X_test_cnn = X_test_padded.reshape(-1, 1, side, side)

# Định dạng lại dữ liệu cho RNN (giả định timestep = 1)
X_train_rnn = X_train.reshape(-1, 1, n_features)  # [samples, timesteps, features]
X_test_rnn = X_test.reshape(-1, 1, n_features)

# Chuyển sang tensor
X_train_cnn_tensor = torch.FloatTensor(X_train_cnn).to(device)
X_train_rnn_tensor = torch.FloatTensor(X_train_rnn).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
X_test_cnn_tensor = torch.FloatTensor(X_test_cnn).to(device)
X_test_rnn_tensor = torch.FloatTensor(X_test_rnn).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

# DataLoader cho CNN và RNN
train_dataset_cnn = TensorDataset(X_train_cnn_tensor, y_train_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
train_dataset_rnn = TensorDataset(X_train_rnn_tensor, y_train_tensor)
train_loader_rnn = DataLoader(train_dataset_rnn, batch_size=64, shuffle=True)

# 1. CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # Tính kích thước đầu ra sau conv và pool
        self.flatten_size = 32 * (side // 4) * (side // 4)
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch_size, timesteps, hidden_size]
        out = self.dropout(out[:, -1, :])  # Lấy output của timestep cuối
        out = self.fc(out)
        return out

# Khởi tạo mô hình
cnn_model = CNNClassifier(num_classes).to(device)
rnn_model = RNNClassifier(input_size=n_features, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)

# Thiết lập training với lịch sử đầy đủ (loss và accuracy)
def train_model(model, train_loader, model_name, X_train_tensor, y_train_tensor, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'epoch': [], 'loss': [], 'accuracy': []}  # Lưu lịch sử huấn luyện
    
    print(f"\nStarting training for {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Đảm bảo dữ liệu trên cùng thiết bị
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Tính accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Tính loss và accuracy trung bình cho epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # Lưu lịch sử huấn luyện vào file
    with open(f"Resuilt/{model_name}_training_history.txt", "w") as f:
        f.write(f"Training History for {model_name}:\n")
        for ep, loss, acc in zip(history['epoch'], history['loss'], history['accuracy']):
            f.write(f"Epoch {ep}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")
    
    # Lưu mô hình sau khi huấn luyện xong
    torch.save(model.state_dict(), f"{model_name.lower()}_ddos_classifier.pth")
    print(f"{model_name} model saved as '{model_name.lower()}_ddos_classifier.pth'")
    
    return history

# Huấn luyện cả hai mô hình
print("Training CNN Model...")
cnn_history = train_model(cnn_model, train_loader_cnn, "CNN", X_train_cnn_tensor, y_train_tensor)
print("\nTraining RNN Model...")
rnn_history = train_model(rnn_model, train_loader_rnn, "RNN", X_train_rnn_tensor, y_train_tensor)

# Đánh giá mô hình
def evaluate_model(model, X_test_tensor, y_test, model_name):
    model.eval()
    X_test_tensor = X_test_tensor.to(device)  # Chuyển dữ liệu sang GPU nếu có
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_tensor, dim=1).cpu().numpy()  # Chuyển về CPU để tính toán
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    target_names = [k for k, v in sorted(attack_mapping.items(), key=lambda x: x[1])]
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Lưu kết quả
    result_text = f"{model_name} Classifier:\nAccuracy: {accuracy:.2f}\nF1-Score: {f1:.2f}\n\n"
    result_text += "Classification Report:\n" + report + "\n"
    with open(f"Resuilt/{model_name}.txt", "w") as f:
        f.write(result_text)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45)
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