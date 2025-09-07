# 🧠 Sentiment Analysis Web Application

Aplikasi web untuk analisis sentimen menggunakan Flask dan model machine learning Ridge Regression. Aplikasi ini dapat menganalisis sentimen dari teks postingan media sosial dengan akurasi tinggi (R² = 0.861).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.0-purple.svg)

## ✨ Fitur Utama

- 🔍 **Analisis Sentimen Real-time**: Prediksi sentimen dari teks dengan respons instan
- 🎯 **Akurasi Tinggi**: Model Ridge Regression dengan R² score 0.861
- 🌐 **Interface Modern**: UI responsif dengan Bootstrap 5 dan desain yang menarik
- 📊 **Visualisasi Hasil**: Tampilan hasil yang informatif dengan chart dan metrik
- 🔧 **API Endpoint**: REST API untuk integrasi dengan sistem lain
- 📱 **Mobile Friendly**: Responsive design untuk semua ukuran layar

## 🚀 Instalasi dan Setup

### Prerequisites

- Python 3.8 atau lebih tinggi
- pip (Python package installer)

### Langkah Instalasi

1. **Clone atau download project ini**

   ```bash
   cd /path/to/your/project
   ```

2. **Buat virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (akan otomatis saat pertama kali menjalankan aplikasi)

   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Pastikan file model tersedia**
   - File `sentiment_model_ridge.joblib` harus ada di direktori project
   - Jika belum ada, jalankan notebook `Final_Project.ipynb` untuk generate model

## 🎮 Cara Menjalankan

### Opsi 1: Production Mode (Recommended)

```bash
python3 run.py
```

Aplikasi akan berjalan di `http://localhost:5000`

### Opsi 2: Development Mode

```bash
python3 app.py --host localhost --port 5000
```

### Opsi 3: Debug Mode

```bash
python3 app.py --host localhost --port 5000 --debug
```

### Opsi 4: Custom Port

```bash
python3 app.py --port 8080
```

### Opsi 5: Production dengan Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📋 Cara Penggunaan

### Web Interface

1. **Buka browser** dan navigasi ke `http://localhost:5000`
2. **Masukkan teks** yang ingin dianalisis di form yang tersedia
3. **Pilih metadata** seperti platform, lokasi, bahasa, dll. (opsional)
4. **Klik "Analisis Sentimen"** untuk mendapatkan hasil
5. **Lihat hasil** berupa skor sentimen, label, dan tingkat keyakinan

### API Usage

#### Endpoint: `POST /api/predict`

**Request Body:**

```json
{
  "text_content": "Saya sangat senang dengan produk ini!",
  "platform": "Twitter",
  "location": "Indonesia",
  "language": "Indonesian",
  "day_of_week": "Monday",
  "mentions": "@brand",
  "hashtags": "#amazing #product"
}
```

**Response:**

```json
{
  "sentiment_score": 0.8245,
  "sentiment_label": "Positive",
  "emotion": "Happy",
  "confidence": 0.649,
  "processed_text": "sangat senang produk"
}
```

#### Contoh dengan curl:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "Produk ini luar biasa!",
    "platform": "Twitter"
  }'
```

#### Contoh dengan Python:

```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "text_content": "Saya kecewa dengan layanan ini",
    "platform": "Facebook",
    "language": "Indonesian"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Sentiment: {result['sentiment_label']} (Score: {result['sentiment_score']})")
```

## 🏗️ Struktur Project

```
final-project/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── sentiment_model_ridge.joblib # Trained model file
├── Final_Project.ipynb        # Jupyter notebook for model training
├── templates/                 # HTML templates
│   ├── base.html             # Base template
│   ├── index.html            # Home page
│   ├── result.html           # Results page
│   ├── about.html            # About page
│   ├── 404.html              # 404 error page
│   └── 500.html              # 500 error page
└── static/                   # Static files
    ├── css/
    │   └── style.css         # Custom CSS
    └── js/
        └── main.js           # Custom JavaScript
```

## 🤖 Tentang Model

### Model Information

- **Algoritma**: Ridge Regression
- **Performance**: R² = 0.861, RMSE = 0.3724, MAE = 0.3118
- **Dataset**: 12,000+ postingan media sosial
- **Features**: TF-IDF vectors (5,000+ features) + metadata
- **Preprocessing**: Text cleaning, tokenization, lemmatization, stemming

### Feature Engineering

- **Text Features**: TF-IDF vectorization dengan 3,000 features
- **Mentions**: TF-IDF vectorization dengan 1,000 features
- **Hashtags**: TF-IDF vectorization dengan 1,000 features
- **Metadata**: Platform, location, language, day of week

## 🎨 Teknologi yang Digunakan

### Backend

- **Flask**: Web framework
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **Pandas & NumPy**: Data manipulation
- **Joblib**: Model serialization

### Frontend

- **Bootstrap 5**: CSS framework
- **Font Awesome**: Icons
- **Google Fonts**: Typography
- **Vanilla JavaScript**: Interactive features

## 📊 Performance Metrics

| Metric        | Value   | Description                 |
| ------------- | ------- | --------------------------- |
| R² Score      | 0.861   | Variance explained by model |
| RMSE          | 0.3724  | Root Mean Square Error      |
| MAE           | 0.3118  | Mean Absolute Error         |
| Training Data | 12,000+ | Number of training samples  |
| Features      | 5,000+  | TF-IDF + metadata features  |

## 🔧 Kustomisasi

### Mengganti Model

1. Train model baru menggunakan notebook yang disediakan
2. Save model dengan nama `sentiment_model_ridge.joblib`
3. Update konfigurasi di `app.py` jika diperlukan

### Menambah Fitur Baru

1. Tambahkan endpoint baru di `app.py`
2. Buat template HTML di folder `templates/`
3. Tambahkan styling di `static/css/style.css`
4. Tambahkan JavaScript di `static/js/main.js`

## 🐛 Troubleshooting

### Aplikasi langsung mati setelah start

Jika aplikasi Flask tiba-tiba berhenti atau crash setelah startup, coba solusi berikut:

**Penyebab Umum:**

- Reloader conflict dalam debug mode
- Host binding issues dengan `0.0.0.0`
- Port sudah digunakan

**Solusi:**

1. **Gunakan production runner** (paling stabil):

   ```bash
   python3 run.py
   ```

2. **Gunakan localhost binding**:

   ```bash
   python3 app.py --host localhost --port 5000
   ```

3. **Debug dengan script khusus**:

   ```bash
   python3 start_app.py
   ```

4. **Disable reloader**:
   ```bash
   python3 app.py --host localhost --port 5000
   ```

### Model tidak dapat dimuat

```
❌ Model file not found. Please ensure 'sentiment_model_ridge.joblib' exists.
```

**Solusi**: Pastikan file model ada di direktori project atau jalankan notebook untuk generate model baru.

### NLTK data tidak ditemukan

```
LookupError: Resource punkt not found.
```

**Solusi**: Download NLTK data manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Port sudah digunakan

```
OSError: [Errno 98] Address already in use
```

**Solusi**: Gunakan port lain atau hentikan proses yang menggunakan port 5000:

```bash
# Cari proses yang menggunakan port 5000
lsof -i :5000

# Hentikan proses
kill -9 <PID>

# Atau gunakan port lain
python app.py --port 8080
```

## 📄 License

Project ini dibuat untuk keperluan pembelajaran dan dapat digunakan secara bebas.

## 👨‍💻 Author

Dibuat dengan ❤️ menggunakan Flask dan Machine Learning

---

**Catatan**: Pastikan Anda memiliki file model `sentiment_model_ridge.joblib` sebelum menjalankan aplikasi. File ini dapat di-generate dengan menjalankan notebook `Final_Project.ipynb`.
