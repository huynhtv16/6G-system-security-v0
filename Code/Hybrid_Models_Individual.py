# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress
import joblib
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Đang sử dụng: {device}")

# # Tạo tensor với dtype float16 để tối ưu tốc độ trên GPU
# x = torch.rand(3, 3, dtype=torch.float16, device=device)
# print(x)

import os
import psutil

num_physical_cores = psutil.cpu_count(logical=False)  # Số lõi vật lý thực tế
print(f"Số lõi vật lý: {num_physical_cores}")

os.environ["LOKY_MAX_CPU_COUNT"] = str(num_physical_cores)


# csv_files = glob.glob('dataset/CSVs/01-12/*.csv')

# data = pd.DataFrame()

# for csv_file in csv_files:
#     print(csv_file)
#     df = pd.read_csv(csv_file)
#     data = pd.concat([data, df])

# data.columns = data.columns.str.strip()

# keys = data.columns.tolist()

# print(keys)

csv_files = glob.glob('dataset/CSVs/01-12/*.csv')

data = pd.DataFrame()

for csv_file in csv_files:
    print(csv_file)
    df = pd.read_csv(csv_file) # encoding='utf-8', low_memory=False

    # Lấy khoảng 1/30 số dòng của mỗi tệp
    df = df.sample(frac=1/300, random_state=42)  

    data = pd.concat([data, df], ignore_index=True)

data.columns = data.columns.str.strip()

keys = data.columns.tolist()
f = open('dataset/keys.txt', 'w')
for key in keys:
    f.write(key + '\n')

# Adjust the error range
error_range = 0.0  # Error range (you can modify this)

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical data (replace 'categorical_column' with your actual categorical column name)
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
data['Source IP'] = data['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
data['Destination IP'] = data['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Convert 'Protocol' column to one-hot encoding (replace 'Protocol' with your actual categorical column name)
data = pd.get_dummies(data, columns=['Protocol'], drop_first=True)

# Remove columns with zero variance (if needed)
# Calculate variances for numeric columns
variances = data.select_dtypes(include=[np.number]).var()

# Identify columns with zero variance
zero_variance_cols = variances[variances == 0].index

# Drop columns with zero variance
data.drop(columns=zero_variance_cols, inplace=True)

# Standardize the data (if needed)
def is_float(x):
    try:
        try:
            float(x)
        except ValueError:
            return False
        return True
    except:
        return False

data.replace([np.inf, -np.inf], [1e10, -1e10], inplace=True)

for column in data.columns:
    if not data[column].apply(is_float).all():
        data = data.drop(column, axis=1)

# Sau khi xử lý data, trước khi split train/test:
# Lưu 1 phần nhỏ dataset vào folder SDB để kiểm tra
os.makedirs('SDB', exist_ok=True)
data_sample = data.sample(frac=1/600, random_state=42)
data_sample.to_csv('SDB/sample_dataset.csv', index=False)

# Lưu danh sách cột features (trừ cột nhãn)
feature_columns = [col for col in data.columns if col != 'Label']
joblib.dump(feature_columns, os.path.join('models', 'feature_columns.pkl'))

# Chuẩn hóa dữ liệu (nếu cần)
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Tách train/test
X = data[feature_columns]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 2: Classifiers
# Naïve Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)  # You can adjust the number of neighbors (k) as needed
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Logistic Regression Classifier
logistic_classifier = LogisticRegression(random_state=42, max_iter=10000)
logistic_classifier.fit(X_train, y_train)
y_pred_lr = logistic_classifier.predict(X_test)

# Tạo thư mục "models" nếu chưa tồn tại
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Lưu các mô hình vào thư mục "models"
joblib.dump(nb_classifier, os.path.join(model_dir, 'naive_bayes_model.pkl'))
joblib.dump(rf_classifier, os.path.join(model_dir, 'random_forest_model.pkl'))
joblib.dump(knn_classifier, os.path.join(model_dir, 'knn_model.pkl'))
joblib.dump(svm_classifier, os.path.join(model_dir, 'svm_model.pkl'))
joblib.dump(logistic_classifier, os.path.join(model_dir, 'logistic_regression_model.pkl'))                       

# Step 3: Evaluation of the Models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model_name, y_test, y_pred, error_range=0.0):
    """
    Đánh giá mô hình và lưu kết quả vào file.
    
    Parameters:
    - model_name (str): Tên mô hình.
    - y_test (array): Nhãn thực tế.
    - y_pred (array): Nhãn dự đoán.
    - error_range (float): Sai số cần trừ vào accuracy.
    """
    # Tính toán accuracy
    accuracy = accuracy_score(y_test, y_pred)
    adjusted_accuracy = accuracy - error_range
    
    # Lưu kết quả vào file txt
    result_text = f"{model_name} Classifier:\n"
    result_text += f"Accuracy: {adjusted_accuracy:.2f}\n\n"
    report = classification_report(y_test, y_pred)
    result_text += "Classification Report:\n" + report + "\n"
    
    with open(f"resuilt/{model_name}.txt", "w") as f:
        f.write(result_text)
    
    # Biểu đồ Accuracy
    plt.figure(figsize=(8, 4))
    plt.barh(['Predicted Accuracy', 'Actual Accuracy'], [adjusted_accuracy, accuracy], color=['blue', 'green'])
    plt.xlim(0, 1.0)
    plt.xlabel('Accuracy')
    plt.title(f'Model Accuracy - {model_name}')
    plt.axvline(x=accuracy, color='green', linestyle='--', label='Prediction Accuracy')
    plt.axvline(x=adjusted_accuracy, color='blue', linestyle='--', label='Actual Accuracy')
    plt.legend()
    plt.savefig(f"images/{model_name}_accuracy.png")
    plt.close()
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred) - error_range
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"images/{model_name}_confusion_matrix.png")
    plt.close()
    
    # Trích xuất precision, recall, f1-score
    lines = report.strip().split('\n')
    data = [line.split() for line in lines[2:]]
    try:
        precision, recall, f1_score, _ = map(float, data[0][1:])
    except ValueError:
        precision, recall, f1_score = 0, 0, 0
    
    # Vẽ biểu đồ các chỉ số
    plt.figure(figsize=(8, 6))
    plt.bar(['Precision', 'Recall', 'F1-Score'], [precision, recall, f1_score], color=['blue', 'green', 'orange'])
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Classification Metrics - {model_name}')
    plt.ylim(0, 1.1)
    plt.savefig(f"images/{model_name}_metrics.png")
    plt.close()

evaluate_model("Naive_Bayes", y_test, y_pred)
evaluate_model("Random_Forest", y_test, y_pred_rf)
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("Logistic_Regression", y_test, y_pred_lr)
