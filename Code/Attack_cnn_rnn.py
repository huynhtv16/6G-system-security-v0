import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ipaddress
import joblib
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
import warnings
warnings.filterwarnings("ignore")

# ==== Cấu hình ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'models'))
CNN_PATH = os.path.join(MODEL_DIR, 'cnn_ddos_classifier.pth')
RNN_PATH = os.path.join(MODEL_DIR, 'rnn_ddos_classifier.pth')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_cnn_rnn.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_columns_cnn_rnn.pkl')
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)
IMAGES_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'images-acc'))
os.makedirs(IMAGES_PATH, exist_ok=True)

attack_mapping = {
    'DrDoS_DNS': 0, 'DrDoS_LDAP': 1, 'DrDoS_MSSQL': 2, 'DrDoS_NetBIOS': 3,
    'DrDoS_NTP': 4, 'DrDoS_SNMP': 5, 'DrDoS_SSDP': 6, 'DrDoS_UDP': 7,
    'Syn': 8, 'TFTP': 9, 'UDPLag': 10
}
attack_mapping_inv = {v: k for k, v in attack_mapping.items()}

# ==== Định nghĩa model ====
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, side):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
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

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# ==== Hàm đánh giá ====
def mean_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def mean_relative_accuracy(y_true, y_pred):
    return 1 - mean_relative_error(y_true, y_pred)

def evaluate_and_save(model, X, y, model_name):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    acc = accuracy_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    mre = mean_relative_error(y, y_pred)
    mra = mean_relative_accuracy(y, y_pred)
    # Lấy tất cả class có thể có
    all_labels = sorted(list(set(y) | set(y_pred)))
    all_names = [attack_mapping_inv[i] for i in all_labels]
    report = classification_report(
        y, y_pred, digits=2, zero_division=0,
        labels=all_labels,
        target_names=all_names
    )
    res = f"=== {model_name} ===\n"
    res += f"Accuracy: {acc:.2f}%\n"
    res += f"F1-score (weighted): {f1:.4f}\n"
    res += f"MRE: {mre:.4f}\n"
    res += f"MRA: {mra:.4f}\n"
    res += report
    with open(os.path.join(RESULTS_DIR, f"{model_name}_result.txt"), "w", encoding="utf-8") as f:
        f.write(res)
    print(res)
    return acc, f1, mre, mra

# ==== Thiết lập logging (ghi ra file và màn hình, UTF-8) ====
LOG_PATH = os.path.join(RESULTS_DIR, 'attack_cnn_rnn.log')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==== Xử lý dữ liệu ====
if len(sys.argv) < 2:
    logging.error("Cách dùng: python Attack_cnn_rnn.py <csv_file>")
    sys.exit(1)
csv_file = sys.argv[1]
logging.info(f"Đang xử lý file: {csv_file}")

# Load scaler và feature columns
try:
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
except Exception as e:
    logging.error(f"Lỗi khi load scaler/feature_columns: {e}")
    sys.exit(1)

# Đọc dữ liệu và xử lý giống Attack_HMI
# Lấy mẫu nhỏ để tránh hết RAM
data = pd.read_csv(csv_file, low_memory=False, skiprows=lambda i: i>0 and np.random.rand() > 1/100000)
data.columns = data.columns.str.strip()
data.dropna(inplace=True)
if 'Label' not in data.columns:
    logging.error("Không tìm thấy cột Label trong dữ liệu.")
    sys.exit(1)
# Giữ lại các attack hợp lệ
data = data[data['Label'].isin(attack_mapping.keys())]
if data.empty:
    logging.warning(f"File {csv_file} không còn mẫu hợp lệ sau khi lọc, bỏ qua.")
    sys.exit(0)
data['Label'] = data['Label'].map(attack_mapping)

if 'Source IP' in data.columns:
    data['Source IP'] = data['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
if 'Destination IP' in data.columns:
    data['Destination IP'] = data['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
if 'Protocol' in data.columns:
    data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

excluded_columns = ['Flow ID', 'Timestamp', 'SimillarHTTP', 'Inbound', 'Label']
X = data.drop(columns=excluded_columns, errors='ignore')
X = X.select_dtypes(include=[np.number])
y = data['Label']

# Đảm bảo đủ feature columns
for col in feature_columns:
    if col not in X.columns:
        X[col] = 0
X = X[feature_columns]
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = scaler.transform(X)
y = y.values

logging.info(f"Số mẫu sau xử lý: {len(data)}")

# ==== Chuẩn bị dữ liệu cho CNN/RNN ====
n_features = X.shape[1]
side = int(np.ceil(np.sqrt(n_features)))
padding = side * side - n_features

# CNN input
X_cnn = np.pad(X, ((0, 0), (0, padding)), mode='constant').reshape(-1, 1, side, side)
X_cnn_tensor = torch.FloatTensor(X_cnn)

# RNN input
X_rnn = X.reshape(-1, 1, n_features)
X_rnn_tensor = torch.FloatTensor(X_rnn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_cnn_tensor = X_cnn_tensor.to(device)
X_rnn_tensor = X_rnn_tensor.to(device)

# ==== Load model và đánh giá ====
num_classes = len(attack_mapping)
cnn_model = CNNClassifier(num_classes, side).to(device)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
rnn_model = RNNClassifier(input_size=n_features, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
rnn_model.load_state_dict(torch.load(RNN_PATH, map_location=device))

cnn_acc, cnn_f1, cnn_mre, cnn_mra = evaluate_and_save(cnn_model, X_cnn_tensor, y, "CNN")
rnn_acc, rnn_f1, rnn_mre, rnn_mra = evaluate_and_save(rnn_model, X_rnn_tensor, y, "RNN")

logging.info(f"Kết quả CNN: Acc={cnn_acc:.2f} F1={cnn_f1:.4f} MRE={cnn_mre:.4f} MRA={cnn_mra:.4f}")
logging.info(f"Kết quả RNN: Acc={rnn_acc:.2f} F1={rnn_f1:.4f} MRE={rnn_mre:.4f} MRA={rnn_mra:.4f}")

# Sau khi fit scaler:
# joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
# joblib.dump(feature_columns, os.path.join(MODEL_DIR, 'feature_columns.pkl'))