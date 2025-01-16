import streamlit as st
import socket

def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        available = True
    except:
        available = False
    sock.close()
    return available

st.title("Bağlantı Testi")

# Port kullanılabilirlik kontrolü
port = 8503
if check_port(port):
    st.success(f"Port {port} kullanılabilir")
else:
    st.error(f"Port {port} kullanımda!")

st.write("Sistem Bilgileri:")
st.json({
    "Python Path": st.runtime.get_instance().python_path,
    "Streamlit Version": st.__version__,
    "Server Address": st.get_option("server.address"),
    "Server Port": st.get_option("server.port")
})
