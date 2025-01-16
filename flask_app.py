from flask import Flask, render_template_string, request, jsonify
import os
from langdetect import detect
from pathlib import Path

app = Flask(__name__)

# HTML şablonu
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Rusça Doküman Arama</title>
    <meta charset="utf-8">
    <style>
        :root {
            --bg-color: #1a1a1a;
            --text-color: #ffffff;
            --border-color: #333;
            --highlight-color: #4a4a4a;
        }
        body { 
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: var(--bg-color);
            color: var(--text-color);
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .sidebar {
            width: 300px;
            padding: 20px;
            background: var(--highlight-color);
            border-radius: 8px;
        }
        .main-content {
            flex: 1;
            padding: 20px;
            background: var(--highlight-color);
            border-radius: 8px;
        }
        .result {
            margin: 10px 0;
            padding: 15px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-color);
        }
        .highlight {
            background: #ffd700;
            color: black;
        }
        input, button {
            background: var(--bg-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            padding: 8px;
            border-radius: 4px;
        }
        button {
            cursor: pointer;
            background: #4CAF50;
            border: none;
        }
        button:hover {
            background: #45a049;
        }
        a {
            color: #4CAF50;
        }
        h1, h2, h3, h4 {
            color: #4CAF50;
        }
    </style>
</head>
<body>
    <h1>🔎 Rusça Doküman Arama Sistemi</h1>
    
    <div class="container">
        <div class="sidebar">
            <h2>📁 Doküman Yükleme</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".txt" multiple>
                <button type="submit">Yükle</button>
            </form>
            
            <h3>📚 Yüklü Dokümanlar</h3>
            <ul id="doc-list">
                {% for doc in documents %}
                    <li>{{ doc }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="main-content">
            <h2>🔍 Arama</h2>
            <form action="/search" method="get">
                <input type="text" name="query" placeholder="Aramak istediğiniz metni girin..." style="width: 70%">
                <button type="submit">Ara</button>
            </form>
            
            <div id="results">
                {% if results %}
                    <h3>Sonuçlar:</h3>
                    {% for result in results %}
                        <div class="result">
                            <h4>{{ result.name }}</h4>
                            <p>{{ result.content }}</p>
                        </div>
                    {% endfor %}
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
"""

class DocumentSearchSystem:
    def __init__(self):
        self.doc_dir = "documents"
        os.makedirs(self.doc_dir, exist_ok=True)
        
    def get_documents(self):
        """Yüklü dokümanların listesini döndür"""
        return [f.name for f in Path(self.doc_dir).glob('*.txt')]
        
    def save_document(self, file) -> bool:
        """Yüklenen dosyayı kaydet"""
        try:
            content = file.read().decode('utf-8')
            if detect(content) == 'ru':
                save_path = os.path.join(self.doc_dir, file.filename)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            return False
        except Exception as e:
            print(f"Hata: {str(e)}")
            return False
            
    def search_documents(self, query: str) -> list:
        """Dokümanlarda arama yap"""
        results = []
        for doc_path in Path(self.doc_dir).glob('*.txt'):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        # Eşleşen bölümü bul ve vurgula
                        start = max(0, content.lower().find(query.lower()) - 50)
                        end = min(len(content), start + 200)
                        snippet = content[start:end]
                        if start > 0:
                            snippet = f"...{snippet}"
                        if end < len(content):
                            snippet = f"{snippet}..."
                            
                        results.append({
                            'name': doc_path.name,
                            'content': snippet
                        })
            except Exception as e:
                print(f"Hata: {str(e)}")
                
        return results

# Global arama sistemi örneği
search_system = DocumentSearchSystem()

@app.route('/')
def home():
    return render_template_string(
        HTML_TEMPLATE,
        documents=search_system.get_documents(),
        results=[]
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
        
    files = request.files.getlist('file')
    success_count = 0
    
    for file in files:
        if file.filename and search_system.save_document(file):
            success_count += 1
            
    return render_template_string(
        HTML_TEMPLATE,
        documents=search_system.get_documents(),
        results=[]
    )

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if query:
        results = search_system.search_documents(query)
    else:
        results = []
        
    return render_template_string(
        HTML_TEMPLATE,
        documents=search_system.get_documents(),
        results=results
    )

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
