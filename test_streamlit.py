import streamlit as st
import time

print("Uygulama başlatılıyor...", flush=True)

try:
    st.title("Test Sayfası")
    st.write("Merhaba, Dünya!")
    st.write("Bu bir test sayfasıdır.")
    
    print("Sayfa yüklendi!", flush=True)
    
except Exception as e:
    print(f"HATA: {str(e)}", flush=True)
    raise e
