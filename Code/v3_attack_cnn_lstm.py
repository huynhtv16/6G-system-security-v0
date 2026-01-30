import os, glob, time, ipaddress, logging, torch
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn

# ================= CONFIG =================
MODEL_PATH = "models-lstm/cnn_best.pth"
DATA_PATH = "dataset/CSVs/01-12/*.csv"
RESULT_DIR = "results-cnn-lstm-test"
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 10000
BATCH_SIZE = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ================= GPU INFO =================
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logging.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    logging.warning("GPU not available — running on CPU")

# ================= LABEL MAP =================
attack_mapping = {
    'DrDoS_DNS': 0,'DrDoS_LDAP': 1,'DrDoS_MSSQL': 2,'DrDoS_NetBIOS': 3,
    'DrDoS_NTP': 4,'DrDoS_SNMP': 5,'DrDoS_SSDP': 6,'DrDoS_UDP': 7,
    'Syn': 8,'TFTP': 9,'UDPLag': 10
}
attack_inv = {v: k for k, v in attack_mapping.items()}

# ================= MODEL =================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, side):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flat = 128*(side//8)*(side//8)
        self.fc = nn.Sequential(
            nn.Linear(self.flat,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

# ================= PREPROCESS =================
def preprocess_chunk(df):
    df.columns = df.columns.str.strip()
    if 'Label' not in df.columns: return pd.DataFrame()
    df = df[df['Label'].isin(attack_mapping.keys())].copy()
    if df.empty: return pd.DataFrame()
    df['Label'] = df['Label'].map(attack_mapping)

    for c in ['Source IP','Destination IP']:
        if c in df.columns:
            try:
                df[c] = df[c].apply(lambda x: int(ipaddress.IPv4Address(x)) if pd.notnull(x) else 0)
            except Exception:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    if 'Protocol' in df.columns:
        df = pd.get_dummies(df, columns=['Protocol'], drop_first=True)
    df = df.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],0).fillna(0)
    return df

# ================= MAIN =================
def main():
    start_all = time.time()
    all_files = sorted(glob.glob(DATA_PATH))
    if not all_files:
        logging.error("No CSV files found!"); return
    logging.info(f"Found {len(all_files)} CSVs.")

    # Fit scaler on first file (define feature_columns)
    scaler = StandardScaler()
    feature_columns = None
    for f in all_files:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = preprocess_chunk(chunk)
            if not chunk.empty:
                X = chunk.drop(columns=['Label'])
                scaler.fit(X)
                feature_columns = list(X.columns)
                n_features = X.shape[1]
                break
        if feature_columns: break

    if not feature_columns:
        logging.error("No valid data found to fit scaler."); return

    side = int(np.ceil(np.sqrt(n_features)))
    model = CNNClassifier(len(attack_mapping), side).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Model loaded on {DEVICE} | features={n_features} | side={side}")

    y_true, y_pred = [], []
    total = 0

    for f in all_files:
        logging.info(f"Processing {f}...")
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = preprocess_chunk(chunk)
            if chunk.empty: continue
            y = chunk['Label'].astype(int).values
            X = chunk.drop(columns=['Label'], errors='ignore')

            # align feature columns
            X = X.reindex(columns=feature_columns, fill_value=0)
            X = scaler.transform(X)
            X_pad = np.pad(X, ((0,0),(0,side*side - X.shape[1])), mode='constant')
            X_tensor = torch.tensor(X_pad.reshape(-1,1,side,side), dtype=torch.float32).to(DEVICE)

            preds = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), BATCH_SIZE):
                    batch = X_tensor[i:i+BATCH_SIZE]
                    out = model(batch)
                    preds.append(out.argmax(1).cpu().numpy())
            preds = np.concatenate(preds)
            y_true.extend(y); y_pred.extend(preds)
            total += len(y)

    # ================= EVAL =================
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    logging.info(f"Accuracy={acc:.4f} | F1={f1:.4f} | Time={time.time()-start_all:.1f}s")

    report = classification_report(y_true, y_pred, target_names=list(attack_mapping.keys()), digits=4)
    with open(os.path.join(RESULT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy={acc:.4f}\nF1={f1:.4f}\n\n{report}")

    logging.info("Report saved -> results-cnn-lstm-test/test_report.txt")

if __name__ == "__main__":
    main()
