from fastapi import FastAPI

# ROUTERS
from .routes import model

app = FastAPI()

app.include_router(model.router, prefix="/model", tags=["model"])


@app.get("/")
async def root():
    return {"message": "Hello World from Main hello"}
