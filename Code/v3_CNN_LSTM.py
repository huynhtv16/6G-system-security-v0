import os, glob, ipaddress, random, logging
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ================= Logging & Config =================
os.makedirs("results-lstm", exist_ok=True)
os.makedirs("images-lstm", exist_ok=True)
MODEL_DIR = "models-lstm"; os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("results-lstm/training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ================= GPU Config =================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("⚠️ GPU not available — training will use CPU (slower).")

# ================= Load dữ liệu mẫu nhỏ =================
csv_files = glob.glob('dataset/CSVs/01-12/*.csv')
data = pd.DataFrame()

def sample_csv(csv_file, sample_frac=1/30):
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for _ in f) - 1
    n_sample = max(1, int(total_lines * sample_frac))
    skip = sorted(random.sample(range(1, total_lines + 1), total_lines - n_sample))
    return pd.read_csv(csv_file, skiprows=skip, low_memory=False)

logging.info("Loading dataset...")
for csv_file in csv_files:
    df = sample_csv(csv_file)
    data = pd.concat([data, df], ignore_index=True)

data.columns = data.columns.str.strip()
data.dropna(inplace=True)

attack_mapping = {
    'DrDoS_DNS': 0,'DrDoS_LDAP': 1,'DrDoS_MSSQL': 2,'DrDoS_NetBIOS': 3,
    'DrDoS_NTP': 4,'DrDoS_SNMP': 5,'DrDoS_SSDP': 6,'DrDoS_UDP': 7,
    'Syn': 8,'TFTP': 9,'UDPLag': 10
}

label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
valid_labels = set(attack_mapping.values())
data = data[data['Label'].isin(valid_labels)]

# IP -> số
data['Source IP'] = data['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
data['Destination IP'] = data['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Protocol one-hot
data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)
data.replace([np.inf, -np.inf], [1e10, -1e10], inplace=True)

# Ép kiểu float
for col in data.columns:
    try:
        data[col] = data[col].astype(float)
    except:
        data.drop(columns=[col], inplace=True)

feature_columns = list(data.drop(columns=['Label']).columns)
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=['Label']))
y = data['Label']

# ================= Split =================
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
n_features = X_train.shape[1]

# ================= Chuẩn bị cho CNN + LSTM =================
side = int(np.ceil(np.sqrt(n_features))); pad = side * side - n_features
X_train_cnn = np.pad(X_train, ((0,0),(0,pad))).reshape(-1,1,side,side)
X_val_cnn = np.pad(X_val, ((0,0),(0,pad))).reshape(-1,1,side,side)
X_test_cnn = np.pad(X_test, ((0,0),(0,pad))).reshape(-1,1,side,side)

X_train_lstm = X_train.reshape(-1,1,n_features)
X_val_lstm = X_val.reshape(-1,1,n_features)
X_test_lstm = X_test.reshape(-1,1,n_features)

# ================= Tensor & DataLoader =================
to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
X_train_cnn_tensor, X_val_cnn_tensor, X_test_cnn_tensor = map(to_tensor,[X_train_cnn,X_val_cnn,X_test_cnn])
X_train_lstm_tensor, X_val_lstm_tensor, X_test_lstm_tensor = map(to_tensor,[X_train_lstm,X_val_lstm,X_test_lstm])
y_train_tensor, y_val_tensor, y_test_tensor = map(lambda x: torch.tensor(x, dtype=torch.long), [y_train.values,y_val.values,y_test.values])

train_loader_cnn = DataLoader(TensorDataset(X_train_cnn_tensor,y_train_tensor), batch_size=256, shuffle=True, pin_memory=True)
val_loader_cnn = DataLoader(TensorDataset(X_val_cnn_tensor,y_val_tensor), batch_size=256, pin_memory=True)
train_loader_lstm = DataLoader(TensorDataset(X_train_lstm_tensor,y_train_tensor), batch_size=256, shuffle=True, pin_memory=True)
val_loader_lstm = DataLoader(TensorDataset(X_val_lstm_tensor,y_val_tensor), batch_size=256, pin_memory=True)

# ================= Model =================
class CNNClassifier(nn.Module):
    def __init__(self,num_classes):
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

class LSTMClassifier(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

cnn_model = CNNClassifier(len(attack_mapping)).to(device)
lstm_model = LSTMClassifier(n_features,256,3,len(attack_mapping)).to(device)

# ================= Train Function =================
def train_model(model, train_loader, val_loader, name, epochs=60, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    best_acc, patience_ct = 0, 0

    for ep in range(epochs):
        model.train(); tr_loss=0; tr_correct=0; total=0
        for X, y in tqdm(train_loader, desc=f"{name} Epoch {ep+1}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X).float()
            y = y.long()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = tr_correct / total

        model.eval(); val_correct = val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X).float()
                y = y.long()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        logging.info(f"{name} Epoch {ep+1}: TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc; patience_ct = 0
            torch.save(model.state_dict(), f"{MODEL_DIR}/{name.lower()}_best.pth")
        else:
            patience_ct += 1
            if patience_ct >= patience:
                logging.info(f"{name}: Early stopping at epoch {ep+1}")
                break

    logging.info(f"{name} done. Best ValAcc={best_acc:.3f}")

# ================= Training =================
train_model(cnn_model, train_loader_cnn, val_loader_cnn, "CNN")
train_model(lstm_model, train_loader_lstm, val_loader_lstm, "LSTM")

# ================= Evaluate =================
cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/cnn_best.pth", map_location=device))
lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_best.pth", map_location=device))
cnn_model.eval(); lstm_model.eval()

with torch.no_grad():
    out_cnn = cnn_model(X_test_cnn_tensor.to(device)).float()
    out_lstm = lstm_model(X_test_lstm_tensor.to(device)).float()
    out_comb = 0.6*out_cnn + 0.4*out_lstm
    preds = out_comb.argmax(1).cpu().numpy()

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')
report = classification_report(y_test, preds, target_names=attack_mapping.keys())
logging.info(f"Ensemble Accuracy: {acc:.3f}, F1={f1:.3f}")
logging.info("\n" + report)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=attack_mapping.keys(), yticklabels=attack_mapping.keys())
plt.title("Confusion Matrix - CNN+LSTM (GPU)")
plt.savefig("images-lstm/ensemble_cm.png", bbox_inches="tight")
plt.close()
