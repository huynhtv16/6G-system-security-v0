import torch
import torch.nn as nn
import math
from torchviz import make_dot

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

        # placeholder, sẽ tính lại sau dựa vào đầu ra thực tế
        self.cnn_flat = None
        self.lstm = nn.LSTM(n_features, 256, 2, batch_first=True, dropout=0.001)
        self.fc = None  # sẽ tạo sau

    def build_fc(self, sample_input, n_classes):
        # chạy 1 lần forward giả để lấy shape thật của cnn output
        with torch.no_grad():
            c_out = self.cnn(sample_input).view(sample_input.size(0), -1)
            self.cnn_flat = c_out.size(1)
        # tạo fully-connected khớp với shape này
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_flat + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

    def forward(self, xc, xl):
        if self.fc is None:
            raise RuntimeError("⚠️ Bạn chưa gọi build_fc() sau khi khởi tạo model.")
        c = self.cnn(xc).view(xc.size(0), -1)
        l, _ = self.lstm(xl)
        l = l[:, -1, :]
        x = torch.cat([c, l], dim=1)
        return self.fc(x)


# ==== Tạo input phù hợp ====
n_features = 200
n_classes = 12
side = int(math.ceil(math.sqrt(n_features)))

model = CNNLSTM_Fusion(n_features, n_classes)

# Input mẫu để build fc
xc = torch.randn(1, 1, side, side)
xl = torch.randn(1, 1, n_features)

# Xây lại fully-connected chính xác với output CNN thực tế
model.build_fc(xc, n_classes)

# Kiểm tra forward
y = model(xc, xl)
print("✅ Output shape:", y.shape)
print("✅ CNN flatten:", model.cnn_flat)

# ==== Vẽ sơ đồ ====
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('cnn_lstm_structure_dynamic')

print("📊 Sơ đồ mô hình đã lưu thành cnn_lstm_structure_dynamic.png")
