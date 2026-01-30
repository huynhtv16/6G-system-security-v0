
"""
Chạy 1 lượt: parse tất cả .txt và xuất 1 file Excel (raw + summary).
"""

import re
from pathlib import Path
import logging
import pandas as pd

# cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

attack_mapping = {
    'DrDoS_DNS': 0, 'DrDoS_LDAP': 1, 'DrDoS_MSSQL': 2, 'DrDoS_NetBIOS': 3,
    'DrDoS_NTP': 4, 'DrDoS_SNMP': 5, 'DrDoS_SSDP': 6, 'DrDoS_UDP': 7,
    'Syn': 8, 'TFTP': 9, 'UDPLag': 10
}

# regex patterns (hỗ trợ giá trị âm)
_PATTERNS = {
    'header': re.compile(r'={2,}\s*([^\s(]+)\s*\(?([A-Za-z0-9_+-]*)\)?\s*={2,}'),
    'total_samples': re.compile(r'Tổng mẫu[:\s]*([0-9,.-]+)', re.IGNORECASE),
    'num_chunks':    re.compile(r'Số chunk[:\s]*([0-9,.-]+)', re.IGNORECASE),
    'accuracy':      re.compile(r'Accuracy(?: TB)?[:\s]*([-–]?[0-9.,]+)%?', re.IGNORECASE),
    'f1':            re.compile(r'F1(?: TB)?[:\s]*([-–]?[0-9.,]+)', re.IGNORECASE),
    'mre':           re.compile(r'MRE(?: TB)?[:\s]*([-–]?[0-9.,]+)', re.IGNORECASE),
    'mra':           re.compile(r'MRA(?: TB)?[:\s]*([-–]?[0-9.,]+)', re.IGNORECASE),
    'total_time':    re.compile(r'Tổng thời gian[:\s]*([-–]?[0-9.,]+)s', re.IGNORECASE),
    # english fallbacks
    'total_samples_en': re.compile(r'Total samples[:\s]*([0-9,.-]+)', re.IGNORECASE),
    'num_chunks_en':    re.compile(r'Number of chunk[s]*[:\s]*([0-9,.-]+)', re.IGNORECASE),
    'accuracy_en':      re.compile(r'Accuracy[:\s]*([-–]?[0-9.,]+)%', re.IGNORECASE),
    'total_time_en':    re.compile(r'Total time[:\s]*([-–]?[0-9.,]+)s', re.IGNORECASE),
}

def _clean_number(s):
    if s is None:
        return None
    s = str(s).strip().replace('%', '')
    s = s.replace('–', '-')  # thay dấu “–” bằng “-”
    # loại bỏ dấu ngăn cách nghìn
    if s.count(',') > 0 and s.count('.') == 0:
        if re.match(r'^-?\d+,\d+$', s):
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    else:
        s = s.replace(',', '')
    try:
        return int(s) if re.match(r'^-?\d+$', s) else float(s)
    except:
        try:
            return float(s)
        except:
            return None

def parse_text(text):
    res = {}
    m = _PATTERNS['header'].search(text)
    if m:
        res['model_raw'] = m.group(1).strip()
        res['attack_raw'] = m.group(2).strip() if m.group(2) else None
    else:
        m2 = re.search(r'([^\s(]+)\s*\(\s*([A-Za-z0-9_+-]+)\s*\)', text)
        if m2:
            res['model_raw'] = m2.group(1).strip()
            res['attack_raw'] = m2.group(2).strip()

    for k, pat in _PATTERNS.items():
        mm = pat.search(text)
        if mm:
            res[k] = mm.group(1).strip()

    # fallbacks
    for a,b in (('total_samples_en','total_samples'),('num_chunks_en','num_chunks'),('accuracy_en','accuracy'),('total_time_en','total_time')):
        if b not in res and a in res:
            res[b] = res[a]

    for nf in ['total_samples','num_chunks','accuracy','f1','mre','mra','total_time']:
        res[nf] = _clean_number(res.get(nf))

    return res

def process_folder(folder: Path):
    rows = []
    files = sorted(folder.glob('*.txt'))
    logging.info(f'Found {len(files)} .txt files in {folder.resolve()}')
    for p in files:
        logging.info(f'Parsing {p.name}')
        txt = p.read_text(encoding='utf-8', errors='ignore')
        parsed = parse_text(txt)
        fname = p.stem
        parts = re.split(r'[_-]', fname, maxsplit=1)
        model_from_name = parts[0] if parts else None
        attack_from_name = parts[1] if len(parts)>1 else None
        model = parsed.get('model_raw') or model_from_name
        attack = parsed.get('attack_raw') or attack_from_name
        attack_id = attack_mapping.get(attack, None)
        rows.append({
            'file': p.name,
            'model': model,
            'attack': attack,
            'attack_id': attack_id,
            'total_samples': parsed.get('total_samples'),
            'num_chunks': parsed.get('num_chunks'),
            'accuracy_percent': parsed.get('accuracy'),
            'f1': parsed.get('f1'),
            'mre': parsed.get('mre'),
            'mra': parsed.get('mra'),
            'total_time_s': parsed.get('total_time'),
        })
    return pd.DataFrame(rows)

def make_summary(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(['model','attack'], dropna=False).agg(
        count_files = ('file','count'),
        mean_accuracy = ('accuracy_percent','mean'),
        median_accuracy = ('accuracy_percent','median'),
        mean_f1 = ('f1','mean'),
        median_f1 = ('f1','median'),
    ).reset_index()
    return agg

def main():
    base_folder = Path(r'E:\MBLab\6G-system-security\results-hmi')
    if not base_folder.exists():
        logging.error(f"❌ Folder '{base_folder}' không tồn tại.")
        return

    all_txt_files = list(base_folder.rglob('*.txt'))
    logging.info(f"🔍 Tìm thấy {len(all_txt_files)} file .txt trong '{base_folder.resolve()}'")

    rows = []
    for p in all_txt_files:
        logging.info(f'📄 Đang đọc: {p}')
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
            parsed = parse_text(text)
            fname = p.stem
            parts = re.split(r'[_-]', fname, maxsplit=1)
            model_from_name = parts[0] if parts else None
            attack_from_name = parts[1] if len(parts) > 1 else None
            model = parsed.get('model_raw') or model_from_name
            attack = parsed.get('attack_raw') or attack_from_name
            attack_id = attack_mapping.get(attack, None)
            rows.append({
                'folder': p.parent.name,
                'file': p.name,
                'model': model,
                'attack': attack,
                'attack_id': attack_id,
                'total_samples': parsed.get('total_samples'),
                'num_chunks': parsed.get('num_chunks'),
                'accuracy_percent': parsed.get('accuracy'),
                'f1': parsed.get('f1'),
                'mre': parsed.get('mre'),
                'mra': parsed.get('mra'),
                'total_time_s': parsed.get('total_time'),
            })
        except Exception as e:
            logging.warning(f'⚠️ Lỗi khi xử lý {p}: {e}')

    df = pd.DataFrame(rows)

    # 👉 Sắp xếp: model -> num_chunks (tăng dần)
    df = df.sort_values(by=['model', 'num_chunks'], ascending=[True, True], na_position='last')

    # 👉 Tính số chunk thực tế (hiệu giữa giá trị hiện tại và trước đó trong cùng model)
    df['chunk_real'] = df.groupby('model')['num_chunks'].diff().fillna(df['num_chunks'])

    summary = make_summary(df)

    out = Path('results_all_models.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as ew:
        df.to_excel(ew, sheet_name='raw', index=False)
        summary.to_excel(ew, sheet_name='summary', index=False)

    logging.info(f'✅ Đã ghi {len(df)} dòng vào {out.resolve()} (sheets: raw, summary, có thêm cột chunk_real)')

if __name__ == '__main__':
    main()
