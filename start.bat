@echo off
echo Rusca Dokuman Arama Sistemi baslatiliyor...
cd /d %~dp0

:: Onceki surecleri temizle
taskkill /F /IM pythonw.exe /T 2>nul
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

:: Virtual environment aktif et ve Streamlit'i baslat
call venv\Scripts\activate.bat
start /B streamlit run app.py

:: Biraz bekle ve tarayiciyi ac
timeout /t 5 /nobreak >nul
start http://localhost:8501
