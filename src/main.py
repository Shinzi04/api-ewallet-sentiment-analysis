from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ROUTERS
from .routes import model
from .predictions_model.predict import run_hourly_sentiment

app = FastAPI(
    title="NLP API",
    description="NLP API",
    version="1.0.0",
)

scheduler = BackgroundScheduler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model.router, prefix="/model", tags=["model"])


@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(
        run_hourly_sentiment,
        trigger=IntervalTrigger(minutes=2),
        kwargs={"count": 100},
        id="sentiment_job",
        replace_existing=True
    )

    scheduler.start()
    print(">> Scheduler started! Job will run every 2 minutes.")


@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()
    print(">> Scheduler shut down.")


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
