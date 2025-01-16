@echo off
:: Yönetici hakları iste
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo Yonetici haklari isteniyor...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

echo Rusca Dokuman Arama Sistemi baslatiliyor...

:: Port 8501'i kontrol et ve kullanımdaysa süreci sonlandır
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501"') do (
    taskkill /F /PID %%a 2>nul
)

:: Biraz bekle
timeout /t 2 /nobreak

:: Virtual environment'ı aktive et
call venv\Scripts\activate.bat

:: Streamlit'i başlat
echo Streamlit baslatiliyor...
streamlit run app.py --server.port 8501 --server.address localhost

pause
