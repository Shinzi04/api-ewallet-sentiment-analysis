import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .utils.scraper import PlayStoreScraper

# ========== PATH CONFIG ==========
BASE_DIR = Path(__file__).resolve().parent  # .../src/predictions_model
ARTIFACTS_PATH = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data"
LOG_PATH = DATA_PATH / "predictions" / "hourly_sentiment_log.json"
MODEL_PATH = ARTIFACTS_PATH / "model.h5"

# ========== LOAD MODEL & ARTIFACTS ==========
print("Loading model from:", MODEL_PATH)  #! REMOVE THIS LATER

model = load_model(MODEL_PATH)

with open(ARTIFACTS_PATH / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(ARTIFACTS_PATH / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open(ARTIFACTS_PATH / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MAXLEN = config["MAXLEN"]
NUM_WORDS = config["NUM_WORDS"]


def clean_text_basic(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(texts, brand_name=None) -> pd.DataFrame:
    if isinstance(texts, str):
        texts = [texts]

    clean_texts = [clean_text_basic(t) for t in texts]

    seq = tokenizer.texts_to_sequences(clean_texts)
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    probs = model.predict(padded, verbose=0)
    pred_idx = probs.argmax(axis=1)
    pred_labels = le.inverse_transform(pred_idx)

    df = pd.DataFrame(
        {
            "ulasan": texts,
            "ulasan_clean": clean_texts,
            "pred_label": pred_labels,
            "pred_negatif": probs[:, 0],
            "pred_neutral": probs[:, 1],
            "pred_positif": probs[:, 2],
        }
    )

    if brand_name is not None:
        df["brand"] = brand_name

    return df


def summarize_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby("brand")["pred_label"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
        .reset_index()
    )

    pivot = dist.pivot(index="brand", columns="pred_label", values="percent").fillna(0)
    return pivot.round(1)


def infer_brand_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "ovo" in name:
        return "OVO"
    if "gopay" in name or "gojek" in name:
        return "GoPay"
    if "dana" in name:
        return "DANA"
    return "Unknown"


def load_reviews_from_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["content"] for d in data if d.get("content")]
    scores = [d.get("score") for d in data]
    return texts, scores


def run_hourly_sentiment(count: int = 100):
    now = datetime.now().isoformat(timespec="seconds")
    print(f"[run_hourly_sentiment] START at {now}, count={count}")

    ovo_fetcher = PlayStoreScraper("ovo.id", save_dir=DATA_PATH)
    dana_fetcher = PlayStoreScraper("id.dana", save_dir=DATA_PATH)
    gopay_fetcher = PlayStoreScraper("com.gojek.gopay", save_dir=DATA_PATH)

    REVIEWS_COUNT = count

    ovo_fetcher.run(REVIEWS_COUNT)
    dana_fetcher.run(REVIEWS_COUNT)
    gopay_fetcher.run(REVIEWS_COUNT)

    # === 2. BACA SEMUA JSON DI FOLDER DATA_PATH ===
    all_pred = []
    json_files = sorted(DATA_PATH.glob("*.json"))
    if not json_files:
        print("[run_hourly_sentiment] No JSON files found in", DATA_PATH)
        return

    for path in json_files:
        brand = infer_brand_from_filename(path)
        print(f"[run_hourly_sentiment] Processing {path.name} as brand={brand} ...")

        texts, scores = load_reviews_from_json(path)
        if not texts:
            print(f"[run_hourly_sentiment] No content in {path.name}, skipping.")
            continue

        df_pred = predict(texts, brand_name=brand)

        if len(scores) == len(df_pred):
            df_pred["rating_score"] = scores

        all_pred.append(df_pred)

    if not all_pred:
        print("[run_hourly_sentiment] No predictions generated.")
        return

    df_all = pd.concat(all_pred, ignore_index=True)

    # === 3. SUMMARY PER BRAND ===
    summary = summarize_by_brand(df_all)

    # === 4. SIMPAN HASIL DETAIL & SUMMARY ===
    out_dir = DATA_PATH / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "hourly_predictions_detail.csv"
    summary_path = out_dir / "hourly_sentiment_summary.csv"

    df_all.to_csv(detail_path, index=False)
    summary.to_csv(summary_path)

    summary_reset = summary.reset_index()
    append_hourly_log(summary_reset)

    done = datetime.now().isoformat(timespec="seconds")
    print(f"[run_hourly_sentiment] DONE at {done}")
    print(f"[run_hourly_sentiment] Saved detail -> {detail_path}")
    print(f"[run_hourly_sentiment] Saved summary -> {summary_path}")

    return {
        "summary": summary_reset.to_dict(orient="records"),
        "detail_file": str(detail_path),
        "summary_file": str(summary_path),
    }


def load_latest_summary():
    summary_path = DATA_PATH / "predictions" / "hourly_sentiment_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found at {summary_path}")

    df = pd.read_csv(summary_path)
    if "brand" not in df.columns and df.index.name == "brand":
        df = df.reset_index()

    return df.to_dict(orient="records")


def append_hourly_log(summary_df: pd.DataFrame):
    out_dir = DATA_PATH / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = LOG_PATH

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": summary_df.to_dict(orient="records"),
    }

    try:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as f:
                log_data = json.load(f)
            if not isinstance(log_data, list):
                log_data = []
        else:
            log_data = []
    except Exception:
        log_data = []

    log_data.append(entry)

    if len(log_data) > 24:
        log_data = log_data[-24:]

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def load_hourly_log():
    log_path = LOG_PATH

    if not log_path.exists():
        raise FileNotFoundError(f"Hourly sentiment log not found at {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Hourly sentiment log is in an unexpected format.")

    return data
