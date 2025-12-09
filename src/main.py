from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .predictions_model.predict import run_hourly_sentiment

# ROUTERS
from .routes import model

app = FastAPI(
    title="NLP API",
    description="NLP API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model.router, prefix="/sentiment", tags=["sentiment"])
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


scheduler = BackgroundScheduler()
@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(
        run_hourly_sentiment,
        trigger=IntervalTrigger(hours=1),
        kwargs={"count": 100},
        id="sentiment_job",
        replace_existing=True,
    )
    scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

