@echo off
echo Port ve bağlantı testi başlatılıyor...
"C:/Users/BT/CascadeProjects/rus_doc_search/venv/Scripts/python.exe" -m streamlit run test_connection.py --server.port 8504 --server.address localhost
pause
