@echo off
chcp 65001
REM Kích hoạt môi trường ảo
call .venv\Scripts\activate
cd Code

REM Nhận diện từng file tấn công
for %%F in (..\dataset\CSVs\01-12\*.csv) do (
    echo Đang xử lý %%F ...
    python v3_attack_cnn_lstm.py %%F
)
echo Tất cả các file đã được xử lý.

pause