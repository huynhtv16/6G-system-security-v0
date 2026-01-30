import os
import re
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results-01'))
IMAGES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images-acc-00'))
os.makedirs(IMAGES_PATH, exist_ok=True)

metrics = ['Accuracy', 'F1-score (weighted)', 'MRE', 'MRA']
models = ['CNN', 'RNN']

# attack -> model -> {metric: value}
attack_metrics = {}

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith('_result.txt'):
        continue
    model = None
    for m in models:
        if fname.startswith(m):
            model = m
            break
    if not model:
        continue
    attack = fname.replace(f"{model}_", "").replace("_result.txt", "")
    with open(os.path.join(RESULTS_DIR, fname), encoding='utf-8') as f:
        content = f.read()
    values = {}
    for metric in metrics:
        m = re.search(rf"{re.escape(metric)}:\s*([0-9.]+)", content)
        if m:
            val = float(m.group(1))
            values[metric] = val
    attack_metrics.setdefault(attack, {})[model] = values

attack_names = sorted(attack_metrics.keys())
x = np.arange(len(attack_names))
width = 0.35

for metric in metrics:
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = [attack_metrics.get(attack, {}).get(model, {}).get(metric, 0) for attack in attack_names]
        plt.bar(x + i*width - width/2, vals, width, label=model)
    plt.xticks(x, attack_names, rotation=30, ha='right')
    plt.xlabel('Attack type')
    plt.ylabel(metric)
    plt.title(f'{metric} của CNN và RNN trên từng kiểu tấn công')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_PATH, f"compare_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').lower()}.png"))
    plt.close()
print("Đã vẽ xong các biểu đồ tổng hợp.")