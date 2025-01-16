import streamlit.web.bootstrap
import webbrowser
import threading
import time

def main():
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')

    # Tarayıcıyı aç
    threading.Thread(target=open_browser).start()
    
    # Streamlit'i başlat
    streamlit.web.bootstrap.run(
        "app.py",
        "streamlit run",
        [],
        flag_options={
            'server.port': 5000,
            'server.address': 'localhost'
        }
    )

if __name__ == '__main__':
    main()
