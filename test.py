import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng: {device}")

# Tạo tensor với dtype float16 để tối ưu tốc độ trên GPU
x = torch.rand(3, 3, dtype=torch.float16, device=device)
print(x)

import multiprocessing

num_physical_cores = multiprocessing.cpu_count()  # Tổng số lõi logic (không phân biệt vật lý/hyper-threading)
print(f"Số lõi CPU phát hiện được: {num_physical_cores}")


import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
