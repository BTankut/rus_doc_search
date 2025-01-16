# Rusça Doküman Arama Sistemi

PDF ve TXT formatındaki Rusça dokümanlar içinde arama yapabilen, kullanıcı dostu bir uygulama.

## Özellikler

- PDF ve TXT dosya desteği
- Rusça metin içinde arama
- Dark mode arayüz
- Dosya boyutu ve karakter sayısı istatistikleri
- Arama geçmişi
- Kolay kurulum ve kullanım

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/KULLANICI_ADI/rus_doc_search.git
cd rus_doc_search
```

2. Virtual environment oluşturun:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

1. Masaüstündeki "Rusça Doküman Arama" kısayoluna çift tıklayın
2. PDF veya TXT formatındaki Rusça dokümanlarınızı yükleyin
3. Arama kutusuna Rusça kelime veya cümle yazın
4. Sonuçları inceleyin

## Gereksinimler

- Python 3.8+
- Streamlit
- PyPDF2
- langdetect
- more-itertools

## Lisans

MIT License
