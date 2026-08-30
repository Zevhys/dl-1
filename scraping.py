"""
Script scraping mandiri untuk mengambil data ulasan (review) aplikasi Gojek
dari Google Play Store menggunakan library `google-play-scraper`.

Cara pakai:
    pip install -r requirements.txt
    python scraping.py

Output:
    data/gojek_reviews_raw.csv
"""

import time
import pandas as pd
from google_play_scraper import Sort, reviews

APP_ID = "com.gojek.app"
TARGET_COUNT = 20000
BATCH_SIZE = 200
OUTPUT_PATH = "data/gojek_reviews_raw.csv"


def scrape_reviews(app_id: str, target_count: int, batch_size: int = 200) -> list:
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < target_count:
        result, continuation_token = reviews(
            app_id,
            lang="id",
            country="id",
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token,
        )

        if not result:
            print("Tidak ada review baru yang bisa diambil, menghentikan proses.")
            break

        all_reviews.extend(result)
        print(f"Terkumpul {len(all_reviews)} review...")

        if continuation_token is None:
            break

        time.sleep(1)

    return all_reviews[:target_count]


def main():
    print(f"Mulai scraping review untuk app_id='{APP_ID}', target={TARGET_COUNT} data...")
    raw_reviews = scrape_reviews(APP_ID, TARGET_COUNT, BATCH_SIZE)

    df = pd.DataFrame(raw_reviews)
    df = df[["content", "score", "at", "thumbsUpCount", "reviewCreatedVersion"]]
    df = df.rename(columns={"content": "review", "score": "rating", "at": "date"})
    df = df.dropna(subset=["review"])
    df = df.drop_duplicates(subset=["review"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Selesai. {len(df)} review unik disimpan ke '{OUTPUT_PATH}'.")
    print(df["rating"].value_counts())


if __name__ == "__main__":
    main()
