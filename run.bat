@echo off
setlocal enabledelayedexpansion

REM Kích hoạt môi trường ảo
call .venv\Scripts\activate
cd Code

REM Lặp qua từng file trong thư mục 01-12
for %%F in (..\dataset\CSVs\01-12\*.csv) do (
    set "fname=%%~nF"
    python Attack_HMI.py "%%F" "!fname!"
)

REM Sau khi nhận diện xong thì vẽ biểu đồ
python draw.py

pause