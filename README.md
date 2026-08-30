# Analisis Sentimen Ulasan Aplikasi Gojek (PlayStore)

Submission proyek analisis sentimen (Dicoding) yang menganalisis sentimen
pengguna terhadap aplikasi **Gojek** berdasarkan ulasan (review) yang diambil
dari Google Play Store.

## Struktur Proyek

```
.
├── scraping.py                # Script scraping mandiri (google-play-scraper)
├── requirements.txt           # Dependensi Python
├── data/
│   └── gojek_reviews_raw.csv  # Hasil scraping (dibuat setelah menjalankan scraping.py)
└── notebooks/
    └── analisis_sentimen.ipynb  # Notebook utama: preprocessing, pelabelan,
                                   # ekstraksi fitur, 3 skema pelatihan, evaluasi,
                                   # dan inference/testing
```

## Langkah 1 — Scraping Data (dijalankan mandiri oleh pengguna)

Sandbox CI/agent yang digunakan untuk mengembangkan notebook ini tidak memiliki
akses jaringan ke `play.google.com` (diblokir oleh kebijakan egress), sehingga
scraping **harus dijalankan di mesin lokal Anda atau di Google Colab**.

```bash
pip install -r requirements.txt
python scraping.py
```

Script ini akan mengambil hingga 20.000 review terbaru aplikasi Gojek
(`com.gojek.app`) dan menyimpannya ke `data/gojek_reviews_raw.csv` dengan kolom:
`review`, `rating`, `date`, `thumbsUpCount`, `reviewCreatedVersion`.

Setelah selesai, letakkan file `gojek_reviews_raw.csv` ke folder `data/` pada
repo ini agar bisa dilanjutkan ke tahap preprocessing dan pelatihan model.

## Langkah 2 — Preprocessing, Pelabelan, dan Pelatihan Model

Dilakukan pada `notebooks/analisis_sentimen.ipynb`, mencakup:

1. Cleaning teks & case folding
2. Tokenisasi, stopword removal (Sastrawi), normalisasi
3. Pelabelan sentimen otomatis berdasarkan rating (3 kelas: positif, netral, negatif)
4. Ekstraksi fitur (TF-IDF / Tokenizer+Embedding / Word2Vec)
5. Pelatihan 3 skema model deep learning yang berbeda
6. Evaluasi akurasi (training & testing)
7. Inference/testing dengan bukti output kelas kategorikal
