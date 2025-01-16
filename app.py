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
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# .env dosyasını yükle
load_dotenv()

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
        
        # ChromaDB istemcisini başlat
        try:
            self.chroma_client = PersistentClient(path="chroma_db")
        except Exception as e:
            st.error(f"ChromaDB başlatma hatası: {str(e)}")
            self.chroma_client = None
            return
        
        # Koleksiyonu oluştur veya var olanı al
        try:
            self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='DeepPavlov/rubert-base-cased-sentence'
            )
            
            try:
                self.collection = self.chroma_client.get_collection(
                    name="russian_documents",
                    embedding_function=self.sentence_transformer_ef
                )
            except:
                self.collection = self.chroma_client.create_collection(
                    name="russian_documents",
                    embedding_function=self.sentence_transformer_ef
                )
        except Exception as e:
            st.error(f"Koleksiyon oluşturma hatası: {str(e)}")
            self.collection = None
        
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
            if not content or not detect(content) == 'ru':
                st.warning("⚠️ Bu belge Rusça değil veya metin içermiyor!")
                return False
            
            # Dokümanı parçalara ayır
            chunks = [chunk for chunk in chunked(content.split('\n'), 5) if chunk]
            chunks = ['\n'.join(chunk).strip() for chunk in chunks]
            chunks = [chunk for chunk in chunks if chunk]  # Boş parçaları kaldır
            
            # ChromaDB'ye ekle
            self.collection.add(
                documents=chunks,
                metadatas=[{"title": file.name} for _ in chunks],
                ids=[f"{file.name}_{i}" for i in range(len(chunks))]
            )
            
            # Bellek içi dokümanlara da ekle
            document = {
                'title': file.name,
                'content': content,
                'chunks': chunks,
                'date_added': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.documents.append(document)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Dosya kaydetme hatası: {str(e)}")
            return False
    
    def search_documents(self, query: str, chunk_size: int = 500) -> List[Dict]:
        """Dokümanlarda arama yap"""
        try:
            # Vektör araması
            vector_results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Metin tabanlı arama için regex pattern
            pattern = re.compile(query, re.IGNORECASE)
            
            # Tüm dokümanları tara
            text_results = []
            for doc in self.documents:
                for chunk in doc['chunks']:
                    if pattern.search(chunk):
                        text_results.append({
                            'chunk': chunk,
                            'title': doc['title'],
                            'score': 1.0 if query.lower() in chunk.lower() else 0.8
                        })
            
            # Sonuçları birleştir
            combined_results = []
            
            # Vektör sonuçlarını ekle
            if vector_results['documents']:
                for doc, metadata, score in zip(
                    vector_results['documents'][0],
                    vector_results['metadatas'][0],
                    vector_results['distances'][0]
                ):
                    combined_results.append({
                        'chunk': doc,
                        'title': metadata['title'],
                        'score': 1 - score  # ChromaDB'de düşük mesafe = yüksek benzerlik
                    })
            
            # Metin sonuçlarını ekle
            combined_results.extend(text_results)
            
            # Tekrar eden sonuçları kaldır ve sırala
            seen = set()
            unique_results = []
            for result in combined_results:
                if result['chunk'] not in seen:
                    seen.add(result['chunk'])
                    unique_results.append(result)
            
            # Skora göre sırala
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            
            return unique_results[:5]  # En iyi 5 sonucu döndür
            
        except Exception as e:
            st.error(f"❌ Arama hatası: {str(e)}")
            return []
    
    def highlight_text(self, text: str, query: str) -> str:
        """Metinde arama sorgusunu vurgula"""
        if not query:
            return text
            
        pattern = re.compile(f'({re.escape(query)})', re.IGNORECASE)
        return pattern.sub(r'**\1**', text)
            
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
                        t["result_title"].format(i, result['title'], 1)
                    ):
                        st.markdown(f"""
                        {result['chunk']}
                        
                        ---
                        {t["doc_stats"].format(format_size(len(result['chunk'].encode('utf-8'))), len(result['chunk']))}
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
                    f"Document: {doc['title']}\nContent: {doc['content'][:1000]}"
                    for doc in system.documents
                ])
                
                with st.spinner(t["ai_thinking"]):
                    answer = system.ask_ai(question, all_docs)
                    st.write(answer)

if __name__ == "__main__":
    main()
