# v2_attack_cnn_rnn_fixed.py
import os
import sys
import time
import ipaddress
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- CONFIG ----------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "../models")
RESULTS_DIR = os.path.join(ROOT_DIR, "../results-01")
os.makedirs(RESULTS_DIR, exist_ok=True)

CNN_PATH = os.path.join(MODEL_DIR, "cnn_best.pth")
RNN_PATH = os.path.join(MODEL_DIR, "rnn_best.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

# inference knobs
CSV_CHUNK = 2000          # rows per chunk
CNN_BATCH = 8             # batch size for CNN inference
RNN_BATCH = 128           # batch size for RNN inference
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

attack_mapping = {
    'DrDoS_DNS': 0, 'DrDoS_LDAP': 1, 'DrDoS_MSSQL': 2, 'DrDoS_NetBIOS': 3,
    'DrDoS_NTP': 4, 'DrDoS_SNMP': 5, 'DrDoS_SSDP': 6, 'DrDoS_UDP': 7,
    'Syn': 8, 'TFTP': 9, 'UDPLag': 10
}
attack_inv = {v: k for k, v in attack_mapping.items()}

print(f"Device: {device}  | CSV_CHUNK={CSV_CHUNK}  | CNN_BATCH={CNN_BATCH}  | RNN_BATCH={RNN_BATCH}")

# ---------------- Models ----------------
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, side):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten_size = 128 * max(1, (side // 8)) * max(1, (side // 8))
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size,256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1, self.flatten_size)
        return self.fc(x)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# ---------------- Helpers ----------------
def infer_rnn_num_layers(state_dict, prefix="rnn."):
    max_idx = -1
    for k in state_dict.keys():
        if k.startswith(prefix + "weight_ih_l"):
            suffix = k[len(prefix + "weight_ih_l"):]
            if suffix.isdigit():
                idx = int(suffix)
                if idx > max_idx:
                    max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 1

def prepare_df_features(df_chunk, feature_columns):
    df = df_chunk.copy()
    if 'Source IP' in df.columns:
        df['Source IP'] = df['Source IP'].apply(
            lambda x: int(ipaddress.IPv4Address(x)) if pd.notnull(x) else 0)
    if 'Destination IP' in df.columns:
        df['Destination IP'] = df['Destination IP'].apply(
            lambda x: int(ipaddress.IPv4Address(x)) if pd.notnull(x) else 0)
    if 'Protocol' in df.columns:
        df = pd.get_dummies(df, columns=['Protocol'], drop_first=True)
    # ensure all features present and order matches feature_columns
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def predict_in_batches_model(model, X_tensor, batch_size):
    model.eval()
    soft = nn.Softmax(dim=1)
    probs_list = []
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            out = model(batch)
            probs = soft(out).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            probs_list.append(probs)
            preds_list.append(preds)
    probs = np.vstack(probs_list) if probs_list else np.zeros((0, getattr(model.fc[-1], 'out_features', 0)))
    preds = np.concatenate(preds_list) if preds_list else np.array([], dtype=int)
    return probs, preds

# ---------------- Main ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python v2_attack_cnn_rnn_fixed.py <input_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    if not os.path.exists(input_csv):
        print("Input file not found:", input_csv); sys.exit(1)
    # load scaler and features
    if not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH):
        print("Missing scaler.pkl or features.pkl in models/. Run training first."); sys.exit(1)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print(f"Loaded scaler and {len(feature_columns)} feature columns.")

    # prepare CNN/RNN shapes based on number of features
    n_features = len(feature_columns)
    side = int(np.ceil(np.sqrt(n_features)))
    padding = side * side - n_features
    num_classes = len(attack_mapping)

    # load CNN
    if not os.path.exists(CNN_PATH):
        print("CNN checkpoint not found:", CNN_PATH); sys.exit(1)
    try:
        cnn_state = torch.load(CNN_PATH, map_location=device, weights_only=True)
    except Exception as e:
        print("Failed to load CNN checkpoint:", e); sys.exit(1)
    cnn = CNNClassifier(num_classes=num_classes, side=side).to(device)
    try:
        # handle checkpoints with different dict structures
        if isinstance(cnn_state, dict):
            if 'state_dict' in cnn_state:
                s = cnn_state['state_dict']
            elif 'model_state_dict' in cnn_state:
                s = cnn_state['model_state_dict']
            else:
                s = cnn_state
        else:
            s = cnn_state
        s = {k.replace('module.', ''): v for k, v in s.items()}
        cnn.load_state_dict(s, strict=False)
    except Exception as e:
        print("Failed to load CNN state_dict:", e); sys.exit(1)

    # load RNN
    if not os.path.exists(RNN_PATH):
        print("RNN checkpoint not found:", RNN_PATH); sys.exit(1)
    try:
        raw = torch.load(RNN_PATH, map_location=device, weights_only=True)
    except Exception as e:
        print("Failed to load RNN checkpoint:", e); sys.exit(1)
    if isinstance(raw, dict):
        if 'state_dict' in raw:
            rstate = raw['state_dict']
        elif 'model_state_dict' in raw:
            rstate = raw['model_state_dict']
        else:
            rstate = raw
    else:
        rstate = raw
    rstate = {k.replace('module.', ''): v for k, v in rstate.items()}
    inferred_layers = infer_rnn_num_layers(rstate, prefix="rnn.")
    print("Inferred RNN num_layers =", inferred_layers)
    rnn = RNNClassifier(input_size=n_features, hidden_size=256, num_layers=inferred_layers, num_classes=num_classes).to(device)
    try:
        rnn.load_state_dict(rstate, strict=False)
    except Exception as e:
        print("Loaded RNN with strict=False due to:", e)
        rnn.load_state_dict(rstate, strict=False)

    if device.type == "cpu":
        print("Running on CPU.")

    # streaming read, preprocess, infer chunk by chunk
    results_rows = []
    total_rows = 0
    chunks = pd.read_csv(input_csv, chunksize=CSV_CHUNK, low_memory=False)
    for chunk_idx, df_chunk in enumerate(tqdm(chunks, desc="CSV chunks")):
        start_idx = total_rows
        df_chunk.columns = df_chunk.columns.str.strip()

        # preserve original label column for output, encode to numeric if needed
        if 'Label' not in df_chunk.columns:
            print("Input chunk missing 'Label' column, skipping chunk"); continue

        # filter known attacks if labels are strings
        if df_chunk['Label'].dtype == object or df_chunk['Label'].dtype == str:
            df_chunk = df_chunk[df_chunk['Label'].isin(attack_mapping.keys())].copy()
            if df_chunk.empty:
                continue
            df_chunk['Label_encoded'] = df_chunk['Label'].map(attack_mapping)
        else:
            df_chunk['Label_encoded'] = df_chunk['Label'].astype(int)

        # prepare feature DataFrame aligned with feature_columns
        X_df = prepare_df_features(df_chunk, feature_columns)

        # scaling
        try:
            X_scaled = scaler.transform(X_df)
        except Exception as e:
            if hasattr(scaler, 'feature_names_in_'):
                expected = list(scaler.feature_names_in_)
                X_df = X_df.reindex(columns=expected, fill_value=0.0)
                X_scaled = scaler.transform(X_df)
            else:
                X_scaled = scaler.transform(X_df.values)

        n_chunk = X_scaled.shape[0]
        if n_chunk == 0:
            continue
        X_cnn = np.pad(X_scaled, ((0,0),(0,padding)), mode='constant').reshape(-1,1,side,side)
        X_rnn = X_scaled.reshape(-1,1,n_features)

        X_cnn_t = torch.FloatTensor(X_cnn)
        X_rnn_t = torch.FloatTensor(X_rnn)

        probs_cnn, preds_cnn = predict_in_batches_model(cnn, X_cnn_t, batch_size=CNN_BATCH)
        probs_rnn, preds_rnn = predict_in_batches_model(rnn, X_rnn_t, batch_size=RNN_BATCH)

        topk = min(3, probs_cnn.shape[1]) if probs_cnn.size else 0
        # safe topk handling when no probs available
        if topk > 0:
            cnn_topk = np.sort(probs_cnn, axis=1)[:, ::-1][:, :topk]
            rnn_topk = np.sort(probs_rnn, axis=1)[:, ::-1][:, :topk]
        else:
            cnn_topk = np.zeros((n_chunk, 3))
            rnn_topk = np.zeros((n_chunk, 3))

        for i in range(n_chunk):
            row_idx = start_idx + i
            true_idx = int(df_chunk['Label_encoded'].iloc[i])
            pred_c = int(preds_cnn[i]) if i < len(preds_cnn) else -1
            pred_r = int(preds_rnn[i]) if i < len(preds_rnn) else -1
            results_rows.append({
                'idx': row_idx,
                'true_label_index': true_idx,
                'true_label_name': attack_inv.get(true_idx, str(true_idx)),
                'pred_cnn_index': pred_c,
                'pred_cnn_name': attack_inv.get(pred_c, str(pred_c)),
                'cnn_prob_1': float(cnn_topk[i,0]) if topk>=1 else 0.0,
                'cnn_prob_2': float(cnn_topk[i,1]) if topk>=2 else 0.0,
                'cnn_prob_3': float(cnn_topk[i,2]) if topk>=3 else 0.0,
                'pred_rnn_index': pred_r,
                'pred_rnn_name': attack_inv.get(pred_r, str(pred_r)),
                'rnn_prob_1': float(rnn_topk[i,0]) if topk>=1 else 0.0,
                'rnn_prob_2': float(rnn_topk[i,1]) if topk>=2 else 0.0,
                'rnn_prob_3': float(rnn_topk[i,2]) if topk>=3 else 0.0,
            })

        total_rows += n_chunk

    # ensure dataframe has expected columns even if empty
    expected_cols = [
        'idx','true_label_index','true_label_name',
        'pred_cnn_index','pred_cnn_name','cnn_prob_1','cnn_prob_2','cnn_prob_3',
        'pred_rnn_index','pred_rnn_name','rnn_prob_1','rnn_prob_2','rnn_prob_3'
    ]
    out_df = pd.DataFrame(results_rows, columns=expected_cols)

    # write combined and per-model CSVs (headers will exist even if no rows)
    out_df.to_csv(os.path.join(RESULTS_DIR, "combined_per_sample.csv"), index=False, encoding='utf-8')
    cnn_df = out_df[['idx','true_label_index','true_label_name','pred_cnn_index','pred_cnn_name','cnn_prob_1','cnn_prob_2','cnn_prob_3']]
    rnn_df = out_df[['idx','true_label_index','true_label_name','pred_rnn_index','pred_rnn_name','rnn_prob_1','rnn_prob_2','rnn_prob_3']]
    cnn_df.to_csv(os.path.join(RESULTS_DIR, "cnn_per_sample.csv"), index=False, encoding='utf-8')
    rnn_df.to_csv(os.path.join(RESULTS_DIR, "rnn_per_sample.csv"), index=False, encoding='utf-8')

    # compute metrics robustly
    def save_report(y_true, y_pred, name):
        labels = sorted(attack_inv.keys())
        target_names = [attack_inv[i] for i in labels]
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.size == 0:
            # write a placeholder report
            with open(os.path.join(RESULTS_DIR, f"{name}_result_{int(time.time())}.txt"), "w", encoding='utf-8') as f:
                f.write("No valid samples to compute metrics.\n")
            return float('nan'), float('nan')
        # ensure lengths match
        if y_true.shape[0] != y_pred.shape[0]:
            minlen = min(y_true.shape[0], y_pred.shape[0])
            y_true = y_true[:minlen]
            y_pred = y_pred[:minlen]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        # pass labels so target_names length matches
        rep = classification_report(y_true, y_pred,
                                    labels=labels,
                                    target_names=target_names,
                                    zero_division=0)
        with open(os.path.join(RESULTS_DIR, f"{name}_result_{int(time.time())}.txt"), "w", encoding='utf-8') as f:
            f.write(f"Accuracy: {acc:.6f}\nF1-weighted: {f1:.6f}\n\n{rep}")
        return acc, f1

    y_true_vals = out_df['true_label_index'].dropna().astype(int).values
    y_pred_cnn_vals = out_df['pred_cnn_index'].fillna(-1).astype(int).values
    y_pred_rnn_vals = out_df['pred_rnn_index'].fillna(-1).astype(int).values

    cnn_acc, cnn_f1 = save_report(y_true_vals, y_pred_cnn_vals, "cnn")
    rnn_acc, rnn_f1 = save_report(y_true_vals, y_pred_rnn_vals, "rnn")

    summary = pd.DataFrame([
        {"model":"cnn","accuracy":cnn_acc,"f1_weighted":cnn_f1},
        {"model":"rnn","accuracy":rnn_acc,"f1_weighted":rnn_f1}
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False, encoding='utf-8')

    print("Done. Results in", RESULTS_DIR)
