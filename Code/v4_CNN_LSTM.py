import os
import glob
import random
import logging
import ipaddress
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from tqdm import tqdm

# ================= CONFIG =================
ROOT = Path('.')
DATA_GLOB = 'dataset/CSVs/01-12/*.csv'
RAW_CHUNKS_DIR = Path('data/chunk_raw')
RAW_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path('models-v4-1')
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path('results-v4-1')
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# chunk sizing
NUM_RAW_CHUNKS = 20
MINI_CHUNK_SIZE = 100_000
TRAIN_FRAC = 0.8
HOLDOUT_RAW = 5
CHUNKSIZE_STREAM = 100_000

# training hyperparams (tuned for RTX3060 12GB)
BATCH_SIZE = 128
ACCUM_STEPS = 2
EPOCHS_PER_MINICHUNK = 250
LR = 3e-4
WEIGHT_DECAY = 1e-4

# GPU/AMP
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
USE_AMP = USE_CUDA
GPU_MEMORY_FRACTION = 0.85

# checkpoints and artifact paths
BEST_CKPT = MODEL_DIR / 'best_checkpoint.pth'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
FEATURES_PATH = MODEL_DIR / 'feature_columns_final.pkl'
PERIODIC_CKPT_FMT = MODEL_DIR / 'checkpoint_raw_{idx:02d}.pth'

# misc
DELETE_RAW_AFTER = True
SAVE_PERIODIC = True
NUM_WORKERS = 4
EARLY_STOPPING = False
MAX_NO_IMPROVE = 50

# mapping file name -> attack class id (keep 0 reserved for BENIGN)
ATTACK_MAPPING = {
    'DrDoS_DNS': 1,
    'DrDoS_LDAP': 2,
    'DrDoS_MSSQL': 3,
    'DrDoS_NetBIOS': 4,
    'DrDoS_NTP': 5,
    'DrDoS_SNMP': 6,
    'DrDoS_SSDP': 7,
    'DrDoS_UDP': 8,
    'Syn': 9,
    'TFTP': 10,
    'UDPLag': 11
}
# reverse mapping for convenience
REVERSE_ATTACK_MAPPING = {v: k for k, v in ATTACK_MAPPING.items()}
# include benign mapping
REVERSE_ATTACK_MAPPING[0] = 'BENIGN'
NUM_CLASSES = max(ATTACK_MAPPING.values()) + 1  # 0..11

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'training.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info('v4-1 full-chunk label-merge pipeline start')
logging.info('Device: %s | AMP: %s', DEVICE, USE_AMP)
if DEVICE.type == 'cuda':
    try:
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    except Exception:
        pass
    try:
        logging.info('GPU: %s', torch.cuda.get_device_name(0))
    except Exception:
        pass

# ================ HELPERS ================

def safe_ip_to_int(x):
    try:
        if isinstance(x, str):
            return int(ipaddress.IPv4Address(x))
        elif pd.isna(x):
            return 0
        else:
            return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def infer_label_series(series, fname_label):
    """
    Given a pandas Series of label values (may be textual like 'BENIGN' or 'DrDoS_DNS'),
    return numeric labels where 0 = BENIGN, 1..N = attack classes from ATTACK_MAPPING or filename fallback.
    """
    out = []
    for v in series.tolist():
        if pd.isna(v):
            out.append(fname_label)
            continue
        s = str(v).strip()
        if s == '':
            out.append(fname_label)
            continue
        up = s.upper()
        if 'BENIGN' in up:
            out.append(0)
            continue
        # try to find an attack mapping by substring match
        matched = False
        for name, idx in ATTACK_MAPPING.items():
            if name.upper() in up:
                out.append(idx)
                matched = True
                break
        if matched:
            continue
        # try numeric
        try:
            num = int(float(s))
            # assume 0 means benign
            if num == 0:
                out.append(0)
            elif num in REVERSE_ATTACK_MAPPING:
                out.append(int(num))
            else:
                out.append(fname_label)
        except Exception:
            # fallback to filename label
            out.append(fname_label)
    return pd.Series(out)


def preprocess_df_and_merge_labels(df: pd.DataFrame, fname_label: int, feature_columns=None):
    """
    Preprocess chunk and merge labels properly.
    Returns dataframe numeric with 'Label' column mapped to 0..NUM_CLASSES-1 and only numeric features.
    """
    if df is None or df.shape[0] == 0:
        return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.str.strip()
    # map IPs
    if 'Source IP' in df.columns:
        df['Source IP'] = df['Source IP'].apply(safe_ip_to_int)
    if 'Destination IP' in df.columns:
        df['Destination IP'] = df['Destination IP'].apply(safe_ip_to_int)
    # protocol one-hot
    if 'Protocol' in df.columns:
        try:
            df = pd.get_dummies(df, columns=['Protocol'], drop_first=True)
        except Exception:
            pass
    # ensure label column exists and merge
    if 'Label' in df.columns:
        df['Label'] = infer_label_series(df['Label'], fname_label)
    else:
        # no label column, assign based on filename
        df['Label'] = fname_label

    # ensure numeric
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.replace([np.inf, -np.inf], 0).fillna(0)

    # If user requested specific feature_columns, reindex accordingly
    if feature_columns is not None and len(feature_columns) > 0:
        # keep those features and Label
        cols = [c for c in feature_columns if c in num_df.columns]
        # if some requested features missing, they'll be filled later when reindexing at train time
        num_df = num_df.reindex(columns=[*feature_columns, 'Label'], fill_value=0)
    return num_df


def discover_feature_set(files, sample_rows=50000):
    features = set()
    label_counts = Counter()
    for f in files:
        fname = Path(f).stem
        if fname not in ATTACK_MAPPING:
            logging.warning('File %s not in mapping, skip', f)
            continue
        lbl = ATTACK_MAPPING[fname]
        try:
            df = pd.read_csv(f, nrows=sample_rows, low_memory=False)
        except Exception:
            try:
                reader = pd.read_csv(f, chunksize=sample_rows, low_memory=False)
                df = next(reader, pd.DataFrame())
            except Exception:
                continue
        if df is None or df.shape[0] == 0:
            continue
        proc = preprocess_df_and_merge_labels(df, lbl)
        if proc.empty:
            continue
        feats = [c for c in proc.columns if c != 'Label']
        features.update(feats)
        label_counts.update(proc['Label'].astype(int).tolist())
    logging.info('Discovered %d features', len(features))
    logging.info('Label counts sample: %s', dict(label_counts))
    return sorted(features)


def create_raw_chunks(file_paths, out_dir: Path, num_raw_chunks=NUM_RAW_CHUNKS, chunksize_stream=CHUNKSIZE_STREAM, block_size=1000):
    logging.info('Counting total rows across all files...')
    total = 0
    for p in file_paths:
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                total += sum(1 for _ in f) - 1
        except Exception:
            pass
    if total <= 0:
        raise SystemExit('No data found in CSVs')
    target = max(1, total // num_raw_chunks)
    logging.info('Total rows ~ %d, target per raw ~ %d', total, target)

    out_paths = [out_dir / f'raw_{i:02d}.csv' for i in range(num_raw_chunks)]
    counts = [0] * num_raw_chunks
    shuffled = file_paths.copy()
    random.shuffle(shuffled)
    rr = random.randrange(num_raw_chunks)

    for p in shuffled:
        fname = Path(p).stem
        if fname not in ATTACK_MAPPING:
            logging.warning('Skipping unmapped file: %s', p)
            continue
        lbl = ATTACK_MAPPING[fname]
        logging.info('Streaming %s label %d', fname, lbl)
        reader = pd.read_csv(p, chunksize=chunksize_stream, low_memory=False)
        for chunk in reader:
            if chunk is None or chunk.shape[0] == 0:
                continue
            # merge row-level labels using filename label as fallback
            chunk['Label'] = infer_label_series(chunk.get('Label', pd.Series([pd.NA]*len(chunk))), lbl)
            chunk = chunk.sample(frac=1.0, random_state=random.randint(0,1<<30)).reset_index(drop=True)
            L = len(chunk)
            start = 0
            while start < L:
                end = min(start + block_size, L)
                block = chunk.iloc[start:end]
                idx = rr % num_raw_chunks
                mode = 'a' if out_paths[idx].exists() else 'w'
                header = not out_paths[idx].exists()
                block.to_csv(out_paths[idx], mode=mode, header=header, index=False)
                counts[idx] += len(block)
                rr += 1
                start = end
    logging.info('Raw chunks created, sample counts: %s', counts[:min(10, len(counts))])
    return out_paths


class CNNLSTM_Fusion(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        side = int(math.ceil(math.sqrt(n_features)))
        self.pad = side * side - n_features
        self.side = side
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cnn_flat = 64 * (side // 4) * (side // 4)
        self.lstm = nn.LSTM(n_features, 256, 2, batch_first=True, dropout=0.001)
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_flat + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

    def forward(self, xc, xl):
        c = self.cnn(xc).view(xc.size(0), -1)
        l, _ = self.lstm(xl)
        l = l[:, -1, :]
        x = torch.cat([c, l], dim=1)
        return self.fc(x)


def prepare_arrays_and_dataloaders(df_chunk, feature_columns, scaler, class_weights=None):
    df = df_chunk.sample(frac=1.0, random_state=random.randint(0,1<<30)).reset_index(drop=True)
    n = len(df)
    ntr = int(n * TRAIN_FRAC)
    if ntr < 1:
        return None, None, 0, 0
    df_tr = df.iloc[:ntr]
    df_val = df.iloc[ntr:]
    Xtr = df_tr.reindex(columns=feature_columns, fill_value=0).values.astype(np.float32)
    Xv = df_val.reindex(columns=feature_columns, fill_value=0).values.astype(np.float32)
    ytr = df_tr['Label'].values.astype(np.int64)
    yv = df_val['Label'].values.astype(np.int64)

    Xtr = scaler.transform(Xtr)
    Xv = scaler.transform(Xv)

    side = int(math.ceil(math.sqrt(len(feature_columns))))
    pad = side * side - len(feature_columns)
    def pair(X):
        return (np.pad(X, ((0,0),(0,pad))).reshape(-1,1,side,side),
                X.reshape(-1,1,len(feature_columns)))
    Xtr_c, Xtr_l = pair(Xtr)
    Xv_c, Xv_l = pair(Xv)

    train_ds = TensorDataset(torch.tensor(Xtr_c), torch.tensor(Xtr_l), torch.tensor(ytr))
    val_ds = TensorDataset(torch.tensor(Xv_c), torch.tensor(Xv_l), torch.tensor(yv))

    if class_weights is not None and class_weights.sum() > 0:
        # build sampler per-mini-chunk using class_weights
        sample_weights = np.array([class_weights[int(y)] for y in ytr], dtype=np.float32)
        # avoid zero-sum
        if sample_weights.sum() <= 0:
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        else:
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, len(ytr), len(yv)


def fit_scaler_and_count_labels(train_raw, feature_columns):
    logging.info('Fitting StandardScaler on training raw chunks (incremental)')
    scaler = StandardScaler()
    label_counts = Counter()
    total_rows = 0
    for ridx, rawp in enumerate(train_raw, start=1):
        logging.info(' Scanning raw %d/%d for scaler: %s', ridx, len(train_raw), rawp)
        for chunk in pd.read_csv(rawp, chunksize=MINI_CHUNK_SIZE, low_memory=False):
            proc = preprocess_df_and_merge_labels(chunk, fname_label=0, feature_columns=feature_columns)
            # Note: we pass fname_label=0 here because each chunk already has Label column filled during raw chunk creation
            if proc.empty:
                continue
            X = proc[feature_columns].values.astype(np.float32)
            if len(X) > 0:
                try:
                    scaler.partial_fit(X)
                except Exception:
                    scaler.partial_fit(X[:min(len(X), 10000)])
            label_counts.update(proc['Label'].astype(int).tolist())
            total_rows += len(proc)
    logging.info('Scaler fit on ~%d rows. Label counts: %s', total_rows, dict(label_counts))
    return scaler, label_counts


def compute_class_weights(label_counts):
    total = sum(label_counts.values()) or 1
    weights = []
    for i in range(NUM_CLASSES):
        cnt = label_counts.get(i, 0)
        if cnt == 0:
            weights.append(0.0)
        else:
            weights.append(total / (NUM_CLASSES * cnt))
    return np.array(weights, dtype=np.float32)


# ================ MAIN ================

def main():
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise SystemExit('No CSVs found')
    logging.info('Found %d CSV files', len(files))

    feature_columns = discover_feature_set(files, sample_rows=50000)
    if not feature_columns:
        raise SystemExit('No features discovered')
    logging.info('Feature count: %d', len(feature_columns))

    # create raw chunks if needed
    raw_paths = sorted(RAW_CHUNKS_DIR.glob('raw_*.csv'))
    if len(raw_paths) < NUM_RAW_CHUNKS:
        logging.info('Creating raw chunks (%d)...', NUM_RAW_CHUNKS)
        raw_paths = create_raw_chunks(files, RAW_CHUNKS_DIR, NUM_RAW_CHUNKS, CHUNKSIZE_STREAM)
    raw_paths = sorted(raw_paths)
    if len(raw_paths) <= HOLDOUT_RAW:
        raise SystemExit('Not enough raw chunks for holdout')

    train_raw = raw_paths[:-HOLDOUT_RAW]
    holdout_raw = raw_paths[-HOLDOUT_RAW:]
    random.shuffle(train_raw)
    logging.info('Training raw chunks: %d, holdout: %d', len(train_raw), len(holdout_raw))

    # Fit scaler once across training raw chunks and compute global class distribution
    scaler, label_counts = fit_scaler_and_count_labels(train_raw, feature_columns)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    class_weights_arr = compute_class_weights(label_counts)
    class_weights_tensor = None
    if class_weights_arr.sum() > 0:
        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)
        logging.info('Computed class weights (len=%d)', len(class_weights_arr))

    n_features = len(feature_columns)
    model = CNNLSTM_Fusion(n_features, NUM_CLASSES).to(DEVICE)

    # if best checkpoint exists, try to load weights only (safer)
    if BEST_CKPT.exists():
        try:
            # try weights_only to avoid unsafe pickle (newer pytorch)
            try:
                ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=True)
                # if weights_only=True returned a state_dict directly
                if isinstance(ckpt, dict) and 'model_state' not in ckpt and all(isinstance(v, (dict, torch.Tensor)) for v in ckpt.values()):
                    model.load_state_dict(ckpt)
                    logging.info('Loaded state_dict from checkpoint (weights_only=True) for fine-tune')
                else:
                    # fallback: maybe it's a container with 'model_state'
                    st = ckpt.get('model_state', ckpt.get('model_state_dict', None))
                    if st is not None:
                        model.load_state_dict(st)
                        logging.info('Loaded model_state from checkpoint (weights_only=True container)')
                    else:
                        logging.info('Checkpoint loaded (weights_only) but format unexpected; will attempt normal load below')
            except TypeError:
                # older PyTorch might not support weights_only param
                ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
                st = ckpt.get('model_state', ckpt.get('model_state_dict', None))
                if st is not None:
                    model.load_state_dict(st)
                    logging.info('Loaded model_state from checkpoint for fine-tune')
        except Exception as e:
            logging.warning('Failed loading checkpoint safely: %s; will train from scratch', e)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if class_weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()
    scaler_amp = torch.amp.GradScaler(enabled=USE_AMP)

    best_f1 = 0.0
    no_improve = 0

    # estimate minichunks
    total_rows = 0
    for p in train_raw:
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                total_rows += sum(1 for _ in f) - 1
        except Exception:
            pass
    total_est = max(1, total_rows // MINI_CHUNK_SIZE)
    logging.info('Estimated mini-chunks: %d', total_est)

    # TRAIN
    for ridx, rawp in enumerate(train_raw, start=1):
        logging.info('Processing raw %d/%d: %s', ridx, len(train_raw), rawp)
        reader = pd.read_csv(rawp, chunksize=MINI_CHUNK_SIZE, low_memory=False)
        for midx, mini in enumerate(reader, start=1):
            logging.info(' Raw %d mini %d, rows %d', ridx, midx, len(mini))
            # note: raw chunks created earlier already include merged row labels per create_raw_chunks
            proc = preprocess_df_and_merge_labels(mini, fname_label=0, feature_columns=feature_columns)
            if proc.empty:
                logging.info('  mini empty after preprocess, skip')
                continue
            proc = proc.reindex(columns=[*feature_columns, 'Label'], fill_value=0)

            missing = [c for c in feature_columns if c not in mini.columns]
            if missing:
                logging.warning('  Missing %d features in this mini: sample first 5: %s', len(missing), missing[:5])

            train_loader, val_loader, ntr, nval = prepare_arrays_and_dataloaders(proc, feature_columns, scaler, class_weights_arr)
            if train_loader is None:
                logging.info('  no training possible in this mini, skip')
                continue

            model.train()
            for epoch in range(EPOCHS_PER_MINICHUNK):
                optimizer.zero_grad(set_to_none=True)
                pbar = tqdm(train_loader, desc=f'Train R{ridx}M{midx} E{epoch+1}/{EPOCHS_PER_MINICHUNK}', leave=False, ncols=120)
                for i, batch in enumerate(pbar):
                    xc, xl, yb = batch
                    xc = xc.to(DEVICE, non_blocking=True)
                    xl = xl.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                        out = model(xc, xl)
                        loss = criterion(out, yb) / ACCUM_STEPS
                    scaler_amp.scale(loss).backward()
                    if (i + 1) % ACCUM_STEPS == 0:
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                        optimizer.zero_grad(set_to_none=True)
                        pbar.set_postfix({'loss': float(loss.item()*ACCUM_STEPS)})
                pbar.close()

            # validation
            model.eval()
            y_t, y_p = [], []
            with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                for xc, xl, yb in val_loader:
                    xc = xc.to(DEVICE)
                    xl = xl.to(DEVICE)
                    out = model(xc, xl)
                    preds = out.argmax(dim=1)
                    y_t.extend(yb.cpu().tolist())
                    y_p.extend(preds.cpu().tolist())
            val_f1 = f1_score(y_t, y_p, average='weighted', zero_division=0)
            val_acc = accuracy_score(y_t, y_p)
            logging.info('  Raw %d mini %d: val_acc=%.4f f1=%.4f (ntr=%d nval=%d)', ridx, midx, val_acc, val_f1, ntr, nval)

            if val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                no_improve = 0
                ckpt = {'model_state': model.state_dict(), 'feature_columns': feature_columns}
                torch.save(ckpt, BEST_CKPT)
                joblib.dump(scaler, SCALER_PATH)
                logging.info('  Saved best checkpoint f1=%.4f', best_f1)
            else:
                no_improve += 1
                logging.info('  No improvement streak: %d', no_improve)

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # periodic checkpoint
        if SAVE_PERIODIC:
            try:
                torch.save({'model_state': model.state_dict(), 'feature_columns': feature_columns}, PERIODIC_CKPT_FMT.format(idx=ridx))
                logging.info('Saved periodic checkpoint for raw %d', ridx)
            except Exception as e:
                logging.warning('Failed to save periodic checkpoint: %s', e)

        if EARLY_STOPPING and no_improve >= MAX_NO_IMPROVE:
            logging.info('Early stopping triggered')
            break

    # ============ EVALUATE HOLDOUT =============
    logging.info('Evaluating holdout raw chunks...')
    if BEST_CKPT.exists():
        try:
            ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
            st = ckpt.get('model_state', ckpt.get('model_state_dict', None))
            if st is not None:
                model.load_state_dict(st)
                logging.info('Loaded best checkpoint for evaluation')
            else:
                logging.warning('best checkpoint format unexpected; using current model')
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            logging.warning('Failed to load best checkpoint for eval: %s', e)
    else:
        logging.warning('No best checkpoint found, using current model')

    all_true, all_pred = [], []
    for hp in holdout_raw:
        for chunk in pd.read_csv(hp, chunksize=CHUNKSIZE_STREAM, low_memory=False):
            proc = preprocess_df_and_merge_labels(chunk, fname_label=0, feature_columns=feature_columns)
            if proc.empty:
                continue
            proc = proc.reindex(columns=[*feature_columns, 'Label'], fill_value=0)
            X = scaler.transform(proc[feature_columns].values.astype(np.float32))
            y = proc['Label'].values.astype(np.int64)
            side = int(math.ceil(math.sqrt(len(feature_columns))))
            pad = side*side - len(feature_columns)
            Xc = np.pad(X, ((0,0),(0,pad))).reshape(-1,1,side,side)
            Xl = X.reshape(-1,1,len(feature_columns))
            for i in range(0, len(y), BATCH_SIZE):
                xb_c = torch.tensor(Xc[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
                xb_l = torch.tensor(Xl[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    out = model(xb_c, xb_l)
                    preds = out.argmax(dim=1).cpu().tolist()
                all_true.extend(y[i:i+BATCH_SIZE].tolist())
                all_pred.extend(preds)

    if all_true:
        final_acc = accuracy_score(all_true, all_pred)
        final_f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
        logging.info('HOLDOUT FINAL: ACC=%.4f | F1=%.4f', final_acc, final_f1)
        logging.info('\n' + classification_report(all_true, all_pred, digits=4, zero_division=0))
        # confusion matrix
        try:
            cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
            logging.info('Confusion matrix:\n%s', np.array2string(cm, max_line_width=200))
        except Exception:
            pass
    else:
        logging.warning('No holdout data processed')

    # save final artifacts
    torch.save({'model_state': model.state_dict(), 'feature_columns': feature_columns}, MODEL_DIR / 'final_model.pth')
    joblib.dump(scaler, MODEL_DIR / 'scaler_final.pkl')
    joblib.dump(feature_columns, MODEL_DIR / 'feature_columns_final.pkl')
    logging.info('Training done. Best F1=%.4f', best_f1)

    # cleanup raw chunk files
    if DELETE_RAW_AFTER:
        try:
            count = 0
            for f in RAW_CHUNKS_DIR.glob('raw_*.csv'):
                f.unlink(missing_ok=True); count += 1
            if count > 0:
                logging.info('Deleted %d raw chunk files', count)
            if not any(RAW_CHUNKS_DIR.iterdir()):
                RAW_CHUNKS_DIR.rmdir(); logging.info('Removed raw_chunks directory')
        except Exception as e:
            logging.warning('Cleanup error: %s', e)


if __name__ == '__main__':
    main()
