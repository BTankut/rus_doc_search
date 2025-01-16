import os
import streamlit as st
from typing import List, Dict
from langdetect import detect
from more_itertools import chunked
from pathlib import Path
import re
from datetime import datetime
import time
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Sayfa yapılandırması
st.set_page_config(
    page_title="Rusça Doküman Arama",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Dark mode CSS
st.markdown("""
<style>
    /* Dark mode renkleri */
    :root {
        --background-color: #1a1a1a;
        --text-color: #ffffff;
        --accent-color: #4CAF50;
    }
    
    /* Ana stil */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2d2d2d;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: var(--accent-color) !important;
    }
    
    /* Butonlar */
    .stButton>button {
        background-color: var(--accent-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    
    /* Metin girişi */
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: var(--text-color);
        border: 1px solid #444;
    }
    
    /* Dosya yükleme */
    .stUploadButton>button {
        background-color: #2d2d2d;
        color: var(--text-color);
        border: 1px solid #444;
    }
    
    /* Sonuç kartları */
    .stExpander {
        background-color: #2d2d2d;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

class DocumentSearchSystem:
    def __init__(self):
        self.documents: List[Dict] = []
        self.doc_dir = "documents"
        self.search_history = []
        self.model = None
        self.embeddings = {}
        
        # Doküman dizinini oluştur
        os.makedirs(self.doc_dir, exist_ok=True)
        
        # Model yükleniyor mesajı
        if not st.session_state.get('model_loaded', False):
            with st.spinner('🤖 Yapay zeka modeli yükleniyor... (İlk açılışta biraz zaman alabilir)'):
                self.load_model()
                st.session_state.model_loaded = True
    
    def load_model(self):
        """Rusça dil modelini yükle"""
        self.model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Metin için vektör oluştur"""
        return self.model.encode(text, convert_to_tensor=True)
    
    def compute_similarity(self, query_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
        """Benzerlik skorunu hesapla"""
        return torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), 
                                                   text_embedding.unsqueeze(0)).item()
    
    def extract_pdf_text(self, file) -> str:
        """PDF dosyasından metin çıkar"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"❌ PDF okuma hatası: {str(e)}")
            return ""
        
    def save_document(self, file) -> bool:
        try:
            # Dosya uzantısını kontrol et
            file_ext = Path(file.name).suffix.lower()
            
            if file_ext == '.pdf':
                content = self.extract_pdf_text(file)
                if not content:
                    return False
            elif file_ext == '.txt':
                content = file.read().decode('utf-8')
            else:
                st.warning(f"⚠️ {file.name} desteklenmeyen dosya formatı! Sadece .pdf ve .txt dosyaları kabul edilir.")
                return False
            
            # Rusça kontrolü
            if detect(content) == 'ru':
                # Dosyayı kaydet
                save_path = os.path.join(self.doc_dir, file.name)
                
                # PDF ise orijinal dosyayı kopyala
                if file_ext == '.pdf':
                    file.seek(0)  # Dosya işaretçisini başa al
                    with open(save_path, 'wb') as f:
                        f.write(file.read())
                else:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                self.documents.append({
                    'name': file.name,
                    'path': save_path,
                    'content': content,
                    'size': len(content.encode('utf-8')),
                    'char_count': len(content),
                    'type': 'PDF' if file_ext == '.pdf' else 'TXT',
                    'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                return True
            else:
                st.warning(f"⚠️ {file.name} Rusça değil!")
                return False
        except Exception as e:
            st.error(f"❌ Hata: {file.name} dosyası işlenirken hata oluştu - {str(e)}")
            return False
            
    def highlight_text(self, text: str, query: str) -> str:
        """Metinde arama sorgusunu vurgula"""
        if not query:
            return text
            
        pattern = re.compile(f'({re.escape(query)})', re.IGNORECASE)
        return pattern.sub(r'**\1**', text)
            
    def search_documents(self, query: str, chunk_size: int = 500) -> List[Dict]:
        """Dokümanlarda arama yap"""
        if not query.strip():
            return []
            
        # Arama geçmişine ekle
        self.search_history.append({
            'query': query,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.search_history = self.search_history[-5:]  # Son 5 aramayı tut
        
        results = []
        query_embedding = self.get_embedding(query)
        
        for doc in self.documents:
            content = doc['content']
            chunks = list(chunked(content, chunk_size))
            
            for i, chunk in enumerate(chunks):
                chunk_text = ''.join(chunk)
                
                # Chunk'ın vektörünü hesapla veya cache'den al
                chunk_key = f"{doc['name']}_{i}"
                if chunk_key not in self.embeddings:
                    self.embeddings[chunk_key] = self.get_embedding(chunk_text)
                chunk_embedding = self.embeddings[chunk_key]
                
                # Benzerlik skorunu hesapla
                similarity = self.compute_similarity(query_embedding, chunk_embedding)
                
                # Benzerlik skoru 0.5'ten büyükse sonuçlara ekle
                if similarity > 0.5:
                    results.append({
                        'document': doc['name'],
                        'text': chunk_text,
                        'similarity': similarity,
                        'chunk_index': i,
                        'size': doc['size'],
                        'char_count': doc['char_count'],
                        'type': doc.get('type', 'TXT')
                    })
        
        # Benzerlik skoruna göre sırala
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:10]  # En iyi 10 sonucu döndür

def format_size(size_bytes: int) -> str:
    """Boyutu okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def main():
    # Başlık ve versiyon göstergesi
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🔎 Rusça Doküman Arama Sistemi")
    with col2:
        st.markdown("""
        <div style='background-color: #4CAF50; padding: 10px; border-radius: 5px; text-align: center;'>
            <span style='color: white; font-weight: bold;'>📱 Streamlit v1.41.1</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Browser kapatıldığında uygulamayı sonlandır
    if not st.session_state.get("browser_connected"):
        st.session_state.browser_connected = True
        st.session_state.connection_time = time.time()
    
    # Her 5 saniyede bir bağlantıyı kontrol et
    if time.time() - st.session_state.connection_time > 5:
        st.session_state.connection_time = time.time()
        if not st.session_state.browser_connected:
            st.stop()
            os._exit(0)
    
    # Oturum durumunu başlat
    if 'search_system' not in st.session_state:
        st.session_state.search_system = DocumentSearchSystem()
    
    # Sol sidebar
    with st.sidebar:
        st.header("📁 Doküman Yönetimi")
        
        # Dosya yükleme
        uploaded_files = st.file_uploader(
            "Rusça metin dosyalarını yükleyin (.txt, .pdf)",
            type=['txt', 'pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            success_count = 0
            for file in uploaded_files:
                if st.session_state.search_system.save_document(file):
                    success_count += 1
            
            if success_count > 0:
                st.success(f"✅ {success_count} Rusça doküman başarıyla yüklendi!")
        
        # Yüklü dokümanlar
        st.subheader("📚 Yüklü Dokümanlar")
        for doc in st.session_state.search_system.documents:
            st.markdown(f"""
            • **{doc['name']}** ({doc['type']})  
            _{format_size(doc['size'])} | {doc['char_count']} karakter_  
            Yükleme: {doc['upload_time']}
            """)
        
        # Arama geçmişi
        if st.session_state.search_system.search_history:
            st.subheader("🕒 Arama Geçmişi")
            for history in reversed(st.session_state.search_system.search_history[-5:]):
                st.markdown(f"""
                🔍 _{history['query']}_  
                {history['time']}
                """)
    
    # Ana içerik
    st.header("🔍 Arama")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Arama sorgusu (Rusça):",
            placeholder="Aramak istediğiniz metni girin..."
        )
    
    with col2:
        chunk_size = st.number_input(
            "Parça boyutu:",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
    
    if query:
        results = st.session_state.search_system.search_documents(query, chunk_size)
        
        if not results:
            st.warning("⚠️ Sonuç bulunamadı.")
        else:
            st.success(f"✨ {len(results)} sonuç bulundu!")
            
            for i, result in enumerate(results, 1):
                with st.expander(
                    f"📄 Sonuç {i} - {result['document']} "
                    f"(Parça {result['chunk_index'] + 1})"
                ):
                    st.markdown(f"""
                    {result['text']}
                    
                    ---
                    📊 _Doküman boyutu: {format_size(result['size'])} | {result['char_count']} karakter_
                    """)

if __name__ == "__main__":
    main()
