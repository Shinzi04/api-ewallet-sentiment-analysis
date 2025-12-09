import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from google_play_scraper import Sort, reviews


class PlayStoreScraper:
    def __init__(self, app_id: str, lang="id", country="id", save_dir="data"):
        self.app_id = app_id
        self.lang = lang
        self.country = country
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.save_dir / f"reviews_{app_id}.json"

    def fetch_reviews(self, count=10_000, sort=Sort.NEWEST):
        print(f"Fetching {count} reviews for {self.app_id}...")
        reviews_result, continuation_token = reviews(
            self.app_id, lang=self.lang, country=self.country, sort=sort, count=count
        )
        self.reviews_result = reviews_result
        self.continuation_token = continuation_token
        print(f"Fetched {len(reviews_result)} reviews.")
        return reviews_result

    def filter_reviews(self):
        filtered = [
            {
                "content": r.get("content"),
                "score": r.get("score"),
            }
            for r in getattr(self, "reviews_result", [])
        ]
        self.filtered_reviews = filtered
        print(f"Filtered {len(filtered)} reviews.")
        return filtered

    def save_reviews(self):
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(self.filtered_reviews, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.filtered_reviews)} reviews to {self.file_path}")

    def save(self):
        if not hasattr(self, "filtered_reviews"):
            raise RuntimeError("Call filter_reviews() before save().")

        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(self.filtered_reviews, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.filtered_reviews)} reviews to {self.file_path}")

    def print_rating_counts(self):
        if not hasattr(self, "filtered_reviews"):
            raise RuntimeError("No reviews to analyze.")
        rating_counts = Counter(
            r["score"] for r in self.filtered_reviews if r.get("score") is not None
        )
        print("\nRating Distribution:")
        for i in range(1, 6):
            print(f"  Rating {i}: {rating_counts.get(i, 0)}")

    def run(self, count=100, save=False):
        self.fetch_reviews(count)
        self.filter_reviews()
        self.save()
        # self.print_rating_counts()
        print(f"\nCompleted on {datetime.now():%Y-%m-%d %H:%M:%S}")