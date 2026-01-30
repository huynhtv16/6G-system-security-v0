import os
import glob
import random
import pandas as pd
import numpy as np
import ipaddress
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ==============================
# 1️⃣ Cấu hình và mapping nhãn
# ==============================
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

# ==============================
# 2️⃣ Hàm đọc ngẫu nhiên mẫu nhỏ
# ==============================
def sample_csv(path, frac=1/300, chunksize=100000):
    sampled_chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        # Lấy ngẫu nhiên 1/300 số dòng trong mỗi chunk
        sampled = chunk.sample(frac=frac, random_state=42)
        sampled_chunks.append(sampled)
    if sampled_chunks:
        return pd.concat(sampled_chunks, ignore_index=True)
    return pd.DataFrame()

# ==============================
# 3️⃣ Đọc và gộp toàn bộ dữ liệu mẫu
# ==============================
csv_files = glob.glob('dataset/CSVs/01-12/*.csv')
data = pd.DataFrame()

for csv_file in csv_files:
    print(f"🔹 Đang đọc mẫu nhỏ từ {os.path.basename(csv_file)} ...")
    df = sample_csv(csv_file)
    data = pd.concat([data, df], ignore_index=True)

print(f"✅ Tổng số dòng sau khi lấy mẫu: {len(data)}")

# ==============================
# 4️⃣ Tiền xử lý dữ liệu
# ==============================
data.columns = data.columns.str.strip()
data.dropna(inplace=True)

# Lọc theo nhãn hợp lệ
data = data[data['Label'].isin(attack_mapping.keys())]
data['Label'] = data['Label'].map(attack_mapping)

# Chuyển IP → số
if 'Source IP' in data.columns:
    data['Source IP'] = data['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
if 'Destination IP' in data.columns:
    data['Destination IP'] = data['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# One-hot encoding cho cột Protocol
if 'Protocol' in data.columns:
    data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

# Loại bỏ các cột không cần
excluded_columns = ['Flow ID', 'Timestamp', 'SimillarHTTP', 'Inbound', 'Label']
X = data.drop(columns=excluded_columns, errors='ignore')
X = X.select_dtypes(include=[np.number])
y = data['Label']

# Lưu danh sách features
os.makedirs('models-hmi', exist_ok=True)
joblib.dump(X.columns.tolist(), 'models-hmi/feature_columns.pkl')

# ==============================
# 5️⃣ Xử lý NaN / Infinity
# ==============================
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'models-hmi/scaler.pkl')

# ==============================
# 6️⃣ Chia dữ liệu & Huấn luyện
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model_dir = 'models-hmi'
os.makedirs(model_dir, exist_ok=True)

def train_and_save(model, name):
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))
    print(f"✅ Đã lưu model: {name}.pkl")

# Huấn luyện các model cơ bản
train_and_save(GaussianNB(), 'naive_bayes_model')
train_and_save(RandomForestClassifier(random_state=42), 'random_forest_model')
train_and_save(KNeighborsClassifier(n_neighbors=7), 'knn_model')
train_and_save(SVC(kernel='linear', random_state=42), 'svm_model')
train_and_save(LogisticRegression(random_state=42, max_iter=10000), 'logistic_regression_model')

print("\n🎯 Huấn luyện hoàn tất! Models đã lưu trong thư mục 'models-hmi'")
