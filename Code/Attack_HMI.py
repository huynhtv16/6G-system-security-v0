import os, sys, glob, time, joblib, psutil, logging
import pandas as pd
import numpy as np
import ipaddress
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ================== Logging ==================
os.makedirs("hmilog", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('hmilog/attack_hmi.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("=== BẮT ĐẦU script Attack_HMI.py ===")

if len(sys.argv) < 3:
    logging.error("Usage: python Attack_HMI.py <csv_file> <attack_name>")
    sys.exit(1)

csv_file = sys.argv[1]
attack_name = sys.argv[2]
logging.info(f"Đang xử lý file: {csv_file} | Attack: {attack_name}")

# ================== Cấu hình đường dẫn ==================
model_dir = Path('../models-hmi')
output_dir = Path('../results-hmi')
os.makedirs(output_dir, exist_ok=True)

# ================== Load scaler + features ==================
scaler_path = model_dir / 'scaler.pkl'
features_path = model_dir / 'feature_columns.pkl'

if not scaler_path.exists() or not features_path.exists():
    logging.error("Thiếu scaler.pkl hoặc feature_columns.pkl trong models-hmi/")
    sys.exit(1)

scaler = joblib.load(scaler_path)
feature_columns = joblib.load(features_path)

# ================== Load models (bỏ file hỗ trợ) ==================
model_files = [
    f for f in glob.glob(str(model_dir / '*.pkl'))
    if not any(x in f for x in ['scaler.pkl', 'feature_columns.pkl'])
]

if not model_files:
    logging.error("Không tìm thấy model hợp lệ trong models-hmi/")
    sys.exit(1)

# ================== Hàm tính MRE / MRA ==================
def mean_relative_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def mean_relative_accuracy(y_true, y_pred):
    return 1 - mean_relative_error(y_true, y_pred)

# ================== Đọc CSV theo từng phần ==================
CHUNK_SIZE = 100000
logging.info(f"Đang đọc CSV theo từng phần (chunk size = {CHUNK_SIZE})...")
process = psutil.Process(os.getpid())

# ================== Chạy từng model ==================
for model_path in model_files:
    model_name = Path(model_path).stem
    logging.info(f"=== Đang kiểm tra model: {model_name} ===")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logging.error(f"Lỗi khi load {model_name}: {e}")
        continue

    acc_list, f1_list, mre_list, mra_list = [], [], [], []
    total_samples = 0
    ram_log = []
    start_time = time.time()

    # Đọc CSV theo chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk.columns = chunk.columns.str.strip()
        if 'Label' not in chunk.columns:
            continue

        # Lọc theo attack_name
        chunk = chunk.loc[chunk['Label'] == attack_name]
        if chunk.empty:
            continue

        # ========== Tiền xử lý ==========
        if 'Source IP' in chunk.columns:
            chunk.loc[:, 'Source IP'] = chunk['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        if 'Destination IP' in chunk.columns:
            chunk.loc[:, 'Destination IP'] = chunk['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        if 'Protocol' in chunk.columns:
            chunk = pd.get_dummies(chunk, columns=['Protocol'], drop_first=True)

        # Chuyển Label sang số (đồng nhất với lúc train)
        attack_mapping = {
            'DrDoS_DNS': 0, 'DrDoS_LDAP': 1, 'DrDoS_MSSQL': 2,
            'DrDoS_NetBIOS': 3, 'DrDoS_NTP': 4, 'DrDoS_SNMP': 5,
            'DrDoS_SSDP': 6, 'DrDoS_UDP': 7, 'Syn': 8,
            'TFTP': 9, 'UDPLag': 10
        }
        chunk = chunk[chunk['Label'].isin(attack_mapping.keys())]
        chunk.loc[:, 'Label'] = chunk['Label'].map(attack_mapping)

        # Bỏ các cột không dùng
        X = chunk.drop(columns=['Label', 'Flow ID', 'Timestamp', 'SimillarHTTP', 'Inbound'], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        y = chunk['Label']

        # Đảm bảo cùng feature với model
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        X = scaler.transform(X)

        # ====== Dự đoán & tính toán ======
        # Lọc bỏ NaN trong y thật
        mask = y.notna()
        y = y[mask]
        X = X[mask]

        # Nếu rỗng thì bỏ qua chunk
        if len(y) == 0:
            continue

        # Đảm bảo cả y và y_pred là int
        y = y.astype(int)
        y_pred = model.predict(X)
        y_pred = np.array(y_pred, dtype=int)

        # Đảm bảo cùng shape
        if len(y_pred) != len(y):
            logging.warning(f"⚠ Chunk {chunk_idx+1}: độ dài dự đoán != thật ({len(y_pred)} vs {len(y)}), bỏ qua.")
            continue

        # Tính toán
        acc = accuracy_score(y, y_pred) * 100
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        mre = mean_relative_error(y, y_pred)
        mra = mean_relative_accuracy(y, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)
        mre_list.append(mre)
        mra_list.append(mra)
        total_samples += len(X)

        logging.info(f"  Chunk {chunk_idx+1}: {len(X)} mẫu | Acc={acc:.2f}% | F1={f1:.4f}")

    if total_samples == 0:
        logging.warning(f"Không có dữ liệu {attack_name} trong file.")
        continue

    elapsed = time.time() - start_time

    # ====== Trung bình kết quả ======
    acc_mean = np.mean(acc_list)
    f1_mean = np.mean(f1_list)
    mre_mean = np.mean(mre_list)
    mra_mean = np.mean(mra_list)

    result_text = (
        f"=== {model_name} ({attack_name}) ===\n"
        f"Tổng mẫu: {total_samples}\n"
        f"Số chunk: {len(acc_list)}\n"
        f"Accuracy TB: {acc_mean:.2f}%\n"
        f"F1 TB: {f1_mean:.4f}\n"
        f"MRE TB: {mre_mean:.4f}\n"
        f"MRA TB: {mra_mean:.4f}\n"
        f"Tổng thời gian: {elapsed:.2f}s\n"
    )

    result_file = output_dir / f"{model_name}_{attack_name}.txt"
    ram_log_path = output_dir / f"ram_{model_name}_{attack_name}.txt"

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(result_text)
    with open(ram_log_path, 'w', encoding='utf-8') as f:
        for t, ram in ram_log:
            f.write(f"{t:.2f},{ram:.2f}\n")

    logging.info(f"✔ {model_name}: AccTB={acc_mean:.2f}% | F1TB={f1_mean:.4f} | Lưu tại {result_file}")

logging.info("=== HOÀN TẤT Attack_HMI.py ===")
