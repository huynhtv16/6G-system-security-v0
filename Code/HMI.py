# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress
import seaborn as sns
import dask.dataframe as dd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đường dẫn tới các tệp CSV
csv_files = glob.glob('dataset/CSVs/01-12/*.csv')

# Thư mục lưu các tệp Parquet
output_dir = 'dataset/Parquet-01/'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Khởi tạo một DataFrame rỗng từ pandas
data = dd.from_pandas(pd.DataFrame(), npartitions=1)
# Duyệt qua từng tệp CSV và hợp nhất dữ liệu
for csv_file in csv_files:
    print(f"Đang xử lý: {csv_file}")

    # Đọc tệp CSV với dask và xử lý kiểu dữ liệu cụ thể nếu cần
    df = dd.read_csv(csv_file, dtype={'SimillarHTTP': 'object'}, low_memory=False)

    # Loại bỏ khoảng trắng trong tên cột
    df.columns = df.columns.str.strip()

    # Lấy ngẫu nhiên 1/30 dữ liệu từ tệp hiện tại
    sample_fraction = 1 / 30  # Tỷ lệ mẫu ngẫu nhiên
    df_sampled = df.sample(frac=sample_fraction, random_state=42)

    # Hợp nhất dữ liệu từ các tệp CSV
    data = dd.concat([data, df_sampled], axis=0)

keys = data.columns.tolist()

print(keys)

# Giả sử data là Dask DataFrame
# Loại bỏ các hàng có giá trị thiếu và trả về một DataFrame mới
data_cleaned = data.dropna()

# Nếu bạn muốn xử lý một phần dữ liệu cụ thể, có thể sử dụng map_partitions
data_cleaned = data.map_partitions(lambda df: df.dropna())

# Kích hoạt tính toán nếu cần
data_cleaned.compute()  # hoặc lưu vào file sau khi loại bỏ missing values


# Giả sử data là Dask DataFrame
label_encoder = LabelEncoder()

# Xử lý các giá trị thiếu trước khi áp dụng LabelEncoder
data['Label'] = data['Label'].fillna('missing_value')

# Sử dụng map_partitions với meta để chỉ định kiểu dữ liệu đầu ra
data['Label'] = data['Label'].map_partitions(
    lambda col: pd.Series(label_encoder.fit_transform(col)),
    meta=('Label', 'int64')
)

# Chuyển đổi địa chỉ IP thành số nguyên
data['Source IP'] = data['Source IP'].map_partitions(
    lambda col: col.apply(lambda x: int(ipaddress.IPv4Address(x))),
    meta=('Source IP', 'int64')
)

data['Destination IP'] = data['Destination IP'].map_partitions(
    lambda col: col.apply(lambda x: int(ipaddress.IPv4Address(x))),
    meta=('Destination IP', 'int64')
)

# Kích hoạt tính toán nếu cần
data.compute()  # hoặc lưu kết quả


# Hàm chuyển đổi one-hot encoding
def get_dummies_protocol(df):
    return pd.get_dummies(df, columns=['Protocol'], drop_first=True)

# Tạo DataFrame mẫu
data = dd.from_pandas(
    pd.DataFrame({'Protocol': [1, 6, 17], 'Value': [10, 20, 30]}), npartitions=1
)

# Apply get_dummies_protocol to a sample of the data to get the correct columns
sample_data = data.head()  # Take a small sample

# The issue was on this line. sample_data is a pandas DataFrame not a Dask DataFrame
# meta_dummies = get_dummies_protocol(sample_data.compute())
meta_dummies = get_dummies_protocol(sample_data) # Remove .compute()


# Apply one-hot encoding with the updated metadata
data = data.map_partitions(get_dummies_protocol, meta=meta_dummies)

# Function to calculate variances
def calculate_variances(df):
    return df.select_dtypes(include=[np.number]).var()

# Calculate variances on Dask DataFrame using the correct metadata
# The original code was missing information to infer the output's data type and shape
# We need to provide a meta that mimics the structure after applying `calculate_variances`
# which results in a pandas Series
variances = data.map_partitions(calculate_variances, meta=pd.Series([], dtype='float64')).compute()


# Identify columns with zero variance
zero_variance_cols = variances[variances == 0].index.tolist()

# Function to drop columns with zero variance
def drop_zero_variance(df, cols_to_drop):
    return df.drop(columns=cols_to_drop, errors='ignore')

# Update metadata after dropping columns with zero variance
meta_drop = meta_dummies.drop(columns=zero_variance_cols, errors='ignore')

# Update Dask DataFrame with the new metadata
data = data.map_partitions(drop_zero_variance, cols_to_drop=zero_variance_cols, meta=meta_drop)

# Compute the result
result = data.compute()
print(result)
