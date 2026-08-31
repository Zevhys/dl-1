"""
Script preprocessing: mengubah hasil scraping mentah menjadi dataset bersih
yang siap dilatih, sekaligus memberi label sentimen berbasis rating.

Fungsi pembersihan di bawah identik dengan yang digunakan pada
notebooks/analisis_sentimen.ipynb.

Cara pakai:
    pip install -r requirements.txt
    python preprocessing.py

Input : data/gojek_reviews_raw.csv   (hasil scraping.py)
Output: data/gojek_reviews_clean.csv

Catatan: stemming Sastrawi bersifat rule-based per kata sehingga proses ini
memakan waktu sekitar 9 menit untuk ~14.000 ulasan.
"""

import os
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

INPUT_PATH = "data/gojek_reviews_raw.csv"
OUTPUT_PATH = "data/gojek_reviews_clean.csv"

SLANG_DICT = {
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "g": "tidak", "nggak": "tidak",
    "ngga": "tidak", "tdk": "tidak", "tidk": "tidak", "dgn": "dengan", "dg": "dengan",
    "utk": "untuk", "untk": "untuk", "sdh": "sudah", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "blum": "belum", "krn": "karena", "karna": "karena", "krna": "karena",
    "jd": "jadi", "jgn": "jangan", "jangn": "jangan", "tp": "tapi", "tapi2": "tapi",
    "trs": "terus", "trus": "terus", "dr": "dari", "dri": "dari", "sy": "saya", "sya": "saya",
    "gw": "saya", "gue": "saya", "gua": "saya", "aku": "saya", "km": "kamu", "kmu": "kamu",
    "lu": "kamu", "loe": "kamu", "lo": "kamu", "anda2": "anda", "bgt": "banget",
    "bngt": "banget", "bgt2": "banget", "banget2": "banget", "bener": "benar", "bnr": "benar",
    "emg": "memang", "emang": "memang", "gmn": "bagaimana", "gimana": "bagaimana",
    "gmna": "bagaimana", "knp": "kenapa", "kenapa2": "kenapa", "napa": "kenapa",
    "skrg": "sekarang", "skrng": "sekarang", "tolg": "tolong", "tlg": "tolong",
    "mksh": "terima kasih", "makasih": "terima kasih", "thx": "terima kasih",
    "thanks": "terima kasih", "tq": "terima kasih", "sm": "sama", "sma": "sama", "jgk": "juga",
    "jg": "juga", "jga": "juga", "bs": "bisa", "bsa": "bisa", "bisa2": "bisa",
    "gabisa": "tidak bisa", "ngak": "tidak", "kaga": "tidak", "tak": "tidak",
    "gaada": "tidak ada", "gaad": "tidak ada", "ada2": "ada", "pdhl": "padahal",
    "padahal2": "padahal", "sll": "selalu", "sllu": "selalu", "selalu2": "selalu",
    "bnyk": "banyak", "byk": "banyak", "banyk": "banyak", "dpt": "dapat", "dapet": "dapat",
    "dapt": "dapat", "kalo": "kalau", "klo": "kalau", "kl": "kalau", "lg": "lagi",
    "lgi": "lagi", "lagi2": "lagi", "hrs": "harus", "harus2": "harus", "msh": "masih",
    "masi": "masih", "aplikasi2": "aplikasi", "apk": "aplikasi", "aplikasinya": "aplikasi",
    "drivernya": "driver", "drivernya2": "driver", "org": "orang", "orng": "orang",
    "org2": "orang", "spy": "supaya", "biar": "supaya", "biarpun": "walaupun",
    "walau": "walaupun", "walopun": "walaupun", "walaupun2": "walaupun", "sblm": "sebelum",
    "sebelum2": "sebelum", "stlh": "setelah", "stelah": "setelah", "abis": "habis",
    "abiss": "habis", "habiss": "habis", "cepet": "cepat", "cpt": "cepat", "cepet2": "cepat",
    "lambat2": "lambat", "lambt": "lambat", "lemot": "lambat", "lelet": "lambat",
    "parah2": "parah", "parahh": "parah", "bgus": "bagus", "bagus2": "bagus", "baguss": "bagus",
    "jelek2": "jelek", "jlek": "jelek", "jelekk": "jelek", "puas2": "puas", "puasss": "puas",
    "kecewa2": "kecewa", "kecewaa": "kecewa", "sngat": "sangat", "sgt": "sangat",
    "sangat2": "sangat", "mantap2": "mantap", "mantul": "mantap", "mantapp": "mantap",
    "top": "bagus", "the best": "sangat bagus", "recomended": "direkomendasikan",
    "rekomended": "direkomendasikan", "worksite": "aplikasi", "eror": "error", "eror2": "error",
    "ngeleg": "lag", "nge lag": "lag", "ngelag": "lag", "hp": "handphone",
    "hpku": "handphone saya", "hpnya": "handphone", "wkwk": "", "wkwkwk": "", "haha": "",
    "hehe": "", "min": "admin", "cs": "customer service", "gojeknya": "gojek",
    "gojek2": "gojek", "drivernya3": "driver",
}

stemmer = StemmerFactory().create_stemmer()
_stopword_factory = StopWordRemoverFactory()
STOPWORDS = set(_stopword_factory.get_stop_words())
STOPWORDS |= {"nya", "yg", "utk", "dgn", "aja", "sih", "deh", "dong", "lah"}
NEGATIONS = {"tidak", "bukan", "jangan", "tanpa", "belum"}
STOPWORDS -= NEGATIONS

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+|#\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

_stem_cache = {}


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    tokens = [SLANG_DICT.get(tok, tok) for tok in tokens]
    tokens = " ".join(tokens).split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def stem_tokens(text: str) -> str:
    out = []
    for tok in text.split():
        if tok not in _stem_cache:
            _stem_cache[tok] = stemmer.stem(tok)
        stemmed = _stem_cache[tok]
        if stemmed:
            out.append(stemmed)
    return " ".join(out)


def preprocess(text: str) -> str:
    return stem_tokens(clean_text(text))

def preprocess(text: str) -> str:
    return stem_tokens(clean_text(text))


def label_from_rating(rating: int) -> str:
    if rating <= 2:
        return "negatif"
    elif rating == 3:
        return "netral"
    else:
        return "positif"


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Memuat {len(df)} ulasan dari '{INPUT_PATH}'")

    df["cleaned"] = df["review"].astype(str).apply(clean_text)

    # Stemming dilakukan pada teks unik saja agar tidak mengulang pekerjaan
    unique_texts = df["cleaned"].unique()
    print(f"Melakukan stemming pada {len(unique_texts)} teks unik...")
    stem_map = {}
    for i, txt in enumerate(unique_texts, 1):
        stem_map[txt] = stem_tokens(txt)
        if i % 2000 == 0:
            print(f"  {i}/{len(unique_texts)} selesai")

    df["final_text"] = df["cleaned"].map(stem_map)
    df["sentiment"] = df["rating"].apply(label_from_rating)

    before = len(df)
    df = df[df["final_text"].str.strip().str.len() > 0]
    print(f"Membuang {before - len(df)} baris yang kosong setelah pembersihan")

    df = df[["review", "rating", "final_text", "sentiment"]]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Selesai. {len(df)} baris disimpan ke '{OUTPUT_PATH}'")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
