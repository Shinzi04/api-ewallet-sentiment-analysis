import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..predictions_model.predict import load_latest_summary, predict, load_hourly_log
from ..predictions_model.utils.scraper import PlayStoreScraper

router = APIRouter()


@router.get("/wallet", tags=["sentiment"])
def get_hourly_summary():
    try:
        summary = load_latest_summary()
        return {"summary": summary}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Summary not available yet. Scheduler has not produced any data.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallet/log", tags=["sentiment"])
def get_hourly_log():
    try:
        log_data = load_hourly_log()
        return {"history": log_data}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Hourly sentiment log not available yet. Scheduler has not produced any data.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SentenceRequest(BaseModel):
    text: str

@router.post("/sentence", tags=["sentiment"])
def predict_sentence_sentiment(payload: SentenceRequest):
    try:
        df = predict(payload.text)
        row = df.iloc[0]

        response = {
            "text": row["ulasan"],
            "clean_text": row["ulasan_clean"],
            "label": row["pred_label"],
            "probabilities_percent": {
                "negatif": round(float(row["pred_negatif"]) * 100, 2),
                "neutral": round(float(row["pred_neutral"]) * 100, 2),
                "positif": round(float(row["pred_positif"]) * 100, 2),
            },
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AppSentimentRequest(BaseModel):
    app_id: str
    count: int = 100


@router.post("/app", tags=["sentiment"])
def predict_app_sentiment(payload: AppSentimentRequest):
    try:
        scraper = PlayStoreScraper(payload.app_id)

        reviews = scraper.fetch_reviews(payload.count)

        if not reviews:
            raise HTTPException(
                status_code=404, detail=f"No reviews found for app_id={payload.app_id}"
            )

        texts = [r.get("content", "") for r in reviews if r.get("content")]

        df_pred = predict(texts, brand_name=payload.app_id)

        dist = df_pred["pred_label"].value_counts(normalize=True).mul(100).round(2)

        sentiments = {
            "negatif": float(dist.get("negatif", 0.0)),
            "neutral": float(dist.get("neutral", 0.0)),
            "positif": float(dist.get("positif", 0.0)),
        }

        return {
            "app_id": payload.app_id,
            "reviews_count": len(df_pred),
            "sentiment_distribution_percent": sentiments,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
