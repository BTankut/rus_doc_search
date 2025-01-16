from flask import Flask, render_template_string
import webbrowser
import threading
import time

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Sayfası</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>Merhaba! 👋</h1>
    <p>Web sunucusu çalışıyor!</p>
    <hr>
    <p>Bu bir test sayfasıdır.</p>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

def open_browser():
    time.sleep(1.5)  # Sunucunun başlaması için bekle
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(host='localhost', port=5000, debug=True)
