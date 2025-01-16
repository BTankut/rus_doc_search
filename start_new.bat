@echo off
echo Rusca Dokuman Arama Sistemi baslatiliyor...
cd /d "%~dp0"

:: Virtual environment aktif et
call venv\Scripts\activate.bat

:: Streamlit'i baslat
start "" /B "venv\Scripts\streamlit.exe" run app.py

:: Biraz bekle ve tarayiciyi ac
timeout /t 5 /nobreak
start http://localhost:8501
