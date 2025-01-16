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
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import requests

# .env dosyasını yükle
load_dotenv()

# Sayfa yapılandırması
st.set_page_config(
    page_title="Rusça Doküman Arama",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS - Pastel Renkler
st.markdown("""
<style>
    /* Dark mode renkleri - Pastel Tonlar */
    :root {
        --background-color: #1a1a2e;
        --secondary-background: #242438;
        --text-color: #e2e2e2;
        --accent-color: #8b8bbd;
        --accent-color-hover: #9d9dce;
        --border-color: #34344a;
    }
    
    /* Ana stil */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: var(--secondary-background);
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: var(--accent-color) !important;
    }
    
    /* Butonlar */
    .stButton>button {
        background-color: var(--accent-color) !important;
        color: var(--text-color) !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 4px !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: var(--accent-color-hover) !important;
    }
    
    /* Metin girişi */
    .stTextInput>div>div>input {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Dosya yükleme */
    .stUploadButton>button {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Sonuç kartları */
    .stExpander {
        background-color: var(--secondary-background) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Seçim kutusu */
    .stSelectbox>div>div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Divider */
    .stDivider {
        border-color: var(--border-color) !important;
    }
    
    /* Tabs */
    .stTabs {
        background-color: var(--secondary-background) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stTab {
        color: var(--text-color) !important;
    }
    
    .stTab[aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Dil çevirileri
TRANSLATIONS = {
    "tr": {
        "title": "📚 Rusça Doküman Arama Sistemi",
        "description": "PDF ve TXT formatındaki Rusça dokümanları yükleyin ve arama yapın.",
        "upload_title": "📝 Doküman Yükleme",
        "upload_label": "Rusça dokümanları yükleyin (PDF/TXT)",
        "search_history": "🔍 Arama Geçmişi",
        "search_tab": "🔍 Doküman Arama",
        "ai_tab": "🤖 Yapay Zeka Sohbet",
        "search_input": "🔍 Arama yapmak için bir kelime veya cümle girin:",
        "ai_input": "💭 Dokümanlar hakkında bir soru sorun:",
        "no_results": "⚠️ Sonuç bulunamadı.",
        "results_found": "✨ {} sonuç bulundu!",
        "result_title": "📄 Sonuç {} - {} (Parça {})",
        "doc_stats": "📊 Doküman boyutu: {} | {} karakter",
        "upload_first": "⚠️ Önce doküman yüklemelisiniz!",
        "ai_thinking": "🤖 Yapay zeka düşünüyor...",
        "error": "Üzgünüm, bir hata oluştu: {}"
    }
}

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

    def ask_ai(self, question: str, context: str) -> str:
        """GPT-4'e soru sor"""
        try:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                return "API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin."

            print(f"API Anahtarı: {api_key[:5]}...") # API anahtarının ilk 5 karakterini yazdır
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/BTankut/rus_doc_search",
                "X-Title": "Russian Document Search",
                "Content-Type": "application/json"
            }
            
            print(f"Headers: {headers}")
            
            data = {
                "model": "openai/gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": """Sen çok yetenekli bir Rusça doküman analiz asistanısın. Aşağıdaki özelliklere sahipsin:

1. Dil Yetenekleri:
   - Rusça metinleri mükemmel şekilde anlama ve analiz etme
   - Rusça-Türkçe çeviri yapabilme
   - Kullanıcının tercih ettiği dilde yanıt verme
   - Teknik ve akademik Rusça terminolojiye hakimiyet

2. Analiz Yetenekleri:
   - Dokümanların ana fikrini çıkarma
   - Önemli noktaları özetleme
   - Metindeki anahtar kavramları belirleme
   - Bağlamsal ilişkileri kurma
   - Karmaşık fikirleri basitleştirme

3. İletişim Tarzı:
   - Net ve anlaşılır ifadeler kullanma
   - Gerektiğinde detaylı açıklamalar yapma
   - Profesyonel ve saygılı bir ton kullanma
   - Kullanıcı sorularını doğru yorumlama

4. Özel Yetenekler:
   - Rusça dokümanlardan alıntı yapabilme
   - Teknik terimleri açıklayabilme
   - Metinler arası bağlantılar kurabilme
   - Gerektiğinde ek kaynaklara yönlendirme

Verilen bağlamı kullanarak soruları bu yetenekler çerçevesinde yanıtla. Her zaman doğru, güvenilir ve yapıcı bilgiler sun."""
                    },
                    {
                        "role": "user",
                        "content": f"""Bağlam:
{context}

Soru:
{question}

Lütfen yukarıdaki yeteneklerini kullanarak bu soruyu yanıtla."""
                    }
                ]
            }
            
            print(f"Request Data: {json.dumps(data, indent=2)}")

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Text: {response.text}")
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API hatası: {response.status_code} - {response.text}"
            
        except Exception as e:
            return f"Üzgünüm, bir hata oluştu: {str(e)}"

def format_size(size_bytes: int) -> str:
    """Boyutu okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def main():
    # Dil seçimi
    if "lang" not in st.session_state:
        st.session_state.lang = "tr"
        
    lang = st.sidebar.selectbox(
        "🌐 Language / Язык / Dil",
        ["Türkçe"],
        index=["tr"].index(st.session_state.lang)
    )
    
    # Dil kodunu güncelle
    st.session_state.lang = {"Türkçe": "tr"}[lang]
    
    # Çevirileri al
    t = TRANSLATIONS[st.session_state.lang]
    
    st.title(t["title"])
    st.write(t["description"])
    
    system = DocumentSearchSystem()
    
    # Sol sidebar
    with st.sidebar:
        st.header(t["upload_title"])
        uploaded_files = st.file_uploader(
            t["upload_label"],
            type=["txt", "pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                system.save_document(file)
                
        st.divider()
        st.header(t["search_history"])
        for h in system.search_history:
            st.text(f"🕒 {h['time']}\n└ {h['query']}")
    
    # Ana içerik
    tab1, tab2 = st.tabs([t["search_tab"], t["ai_tab"]])
    
    # Arama sekmesi
    with tab1:
        query = st.text_input(t["search_input"])
        
        if query:
            results = system.search_documents(query)
            
            if not results:
                st.warning(t["no_results"])
            else:
                st.success(t["results_found"].format(len(results)))
                
                for i, result in enumerate(results, 1):
                    with st.expander(
                        t["result_title"].format(i, result['document'], result['chunk_index'] + 1)
                    ):
                        st.markdown(f"""
                        {result['text']}
                        
                        ---
                        {t["doc_stats"].format(format_size(result['size']), result['char_count'])}
                        """)
    
    # Yapay Zeka sekmesi
    with tab2:
        if not system.documents:
            st.warning(t["upload_first"])
        else:
            question = st.text_input(t["ai_input"])
            
            if question:
                # Tüm dokümanları birleştir
                all_docs = "\n---\n".join([
                    f"Document: {doc['name']}\nContent: {doc['content'][:1000]}"
                    for doc in system.documents
                ])
                
                with st.spinner(t["ai_thinking"]):
                    answer = system.ask_ai(question, all_docs)
                    st.write(answer)

if __name__ == "__main__":
    main()
