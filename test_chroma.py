import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

st.title("ChromaDB Test Sayfası")

try:
    # ChromaDB istemcisini başlat
    chroma_client = PersistentClient(path="chroma_db")
    st.success("✅ ChromaDB bağlantısı başarılı!")
    
    # Koleksiyonları listele
    collections = chroma_client.list_collections()
    st.write("Mevcut Koleksiyonlar:")
    for collection in collections:
        st.write(f"- {collection.name} (belge sayısı: {collection.count()})")
    
    # Test koleksiyonu oluştur
    if st.button("Test Koleksiyonu Oluştur"):
        try:
            # Embedding fonksiyonunu tanımla
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='DeepPavlov/rubert-base-cased-sentence'
            )
            
            # Test koleksiyonunu oluştur
            test_collection = chroma_client.create_collection(
                name="test_collection",
                embedding_function=sentence_transformer_ef
            )
            
            # Test verilerini ekle
            test_collection.add(
                documents=["Это тестовый документ на русском языке.", 
                          "Второй тестовый документ для проверки."],
                ids=["test1", "test2"]
            )
            
            st.success("✅ Test koleksiyonu başarıyla oluşturuldu!")
            
        except Exception as e:
            st.error(f"Test koleksiyonu oluşturulurken hata: {str(e)}")
    
    # Arama testi
    if st.button("Arama Testi Yap"):
        try:
            test_collection = chroma_client.get_collection(
                name="test_collection",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='DeepPavlov/rubert-base-cased-sentence'
                )
            )
            
            results = test_collection.query(
                query_texts=["тестовый документ"],
                n_results=2
            )
            
            st.write("Arama Sonuçları:")
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                st.write(f"{i+1}. Belge: {doc}")
                st.write(f"   Benzerlik: {1 - distance:.4f}")
                
        except Exception as e:
            st.error(f"Arama testi sırasında hata: {str(e)}")

except Exception as e:
    st.error(f"ChromaDB bağlantı hatası: {str(e)}")
