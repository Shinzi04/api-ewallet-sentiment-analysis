import os

from fastapi import APIRouter, HTTPException

from ..predictions_model.predict import load_latest_summary

router = APIRouter()


@router.get("/", tags=["model"])
async def root():
    return {"message": "Hello World"}


@router.get("/summary", tags=["model"])
def get_hourly_summary():
    try:
        summary = load_latest_summary()
        return {"summary": summary}
    except FileNotFoundError:
        # Scheduler hasn't produced anything yet
        raise HTTPException(
            status_code=503,
            detail="Summary not available yet. Scheduler has not produced any data.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
