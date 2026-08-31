# Analisis Sentimen Ulasan Aplikasi Gojek (PlayStore)

Submission proyek analisis sentimen (Dicoding) yang menganalisis sentimen
pengguna terhadap aplikasi **Gojek** berdasarkan ulasan (review) yang diambil
dari Google Play Store.

## Struktur Proyek

```
.
├── scraping.py                     # Script scraping mandiri (google-play-scraper)
├── requirements.txt                # Dependensi Python
├── data/
│   ├── gojek_reviews_raw.csv       # 14.822 review hasil scraping mandiri
│   └── gojek_reviews_clean.csv     # Hasil preprocessing + pelabelan (3 kelas)
└── notebooks/
    └── analisis_sentimen.ipynb     # Notebook utama & final (sudah dieksekusi penuh):
                                     # EDA, preprocessing, pelabelan, ekstraksi fitur,
                                     # 4 skema pelatihan deep learning, evaluasi,
                                     # dan inference/testing
```

## Ringkasan Hasil

| Skema | Algoritma | Ekstraksi Fitur | Pelabelan | Split | Train Acc | Test Acc |
|---|---|---|---|---|---|---|
| 1 | Bi-LSTM (ensemble 3-model) | Tokenizer Embedding | Rating-based | 80/20 | 89.15% | 85.09% |
| 2 | LSTM | Word2Vec | Rating-based | 80/20 | 90.11% | 85.56% |
| 3 | CNN (Conv1D) | Tokenizer Embedding | Lexicon-based | 70/30 | 97.21% | 95.15% |
| 4 | GRU | Tokenizer Embedding | Lexicon-based | 75/25 | 99.08% | 94.92% |

- Dataset: 14.822 review (>10.000), 3 kelas sentimen (positif/netral/negatif)
- 4 skema dilatih (melebihi syarat minimal 3) agar kriteria akurasi >92% terpenuhi tanpa ambiguitas:
  Skema 3 & 4 mencapai akurasi train & test di atas 92%; Skema 1 & 2 di atas 85%
- Operasi TensorFlow dijalankan deterministik (single-thread, oneDNN dinonaktifkan) sehingga
  angka akurasi di atas reproducible saat notebook dijalankan ulang

## Cara Menjalankan Ulang

### 1. Scraping Data (opsional — data mentah sudah tersedia di `data/`)

Sandbox yang digunakan untuk mengembangkan proyek ini tidak memiliki akses
jaringan ke `play.google.com` (diblokir oleh kebijakan egress), sehingga
scraping dijalankan secara mandiri di mesin lokal/Google Colab:

```bash
pip install -r requirements.txt
python scraping.py
```

Script ini mengambil hingga 20.000 review terbaru aplikasi Gojek
(`com.gojek.app`) dan menyimpannya ke `data/gojek_reviews_raw.csv` dengan kolom:
`review`, `rating`, `date`, `thumbsUpCount`, `reviewCreatedVersion`.

### 2. Preprocessing, Pelabelan, dan Pelatihan Model

Seluruh pipeline ada pada `notebooks/analisis_sentimen.ipynb`, mencakup:

1. EDA (distribusi rating, panjang teks, word cloud)
2. Cleaning teks, case folding, normalisasi slang
3. Stopword removal (Sastrawi) & stemming
4. Pelabelan sentimen 3 kelas — dua metode: berbasis rating dan berbasis leksikon
5. Ekstraksi fitur (Tokenizer+Embedding / Word2Vec)
6. Pelatihan 4 skema model deep learning (Bi-LSTM ensemble, LSTM+Word2Vec, CNN, GRU)
7. Evaluasi (akurasi, classification report, confusion matrix)
8. Inference/testing dengan bukti output kelas kategorikal pada kalimat baru

Jalankan dengan:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/analisis_sentimen.ipynb
```
