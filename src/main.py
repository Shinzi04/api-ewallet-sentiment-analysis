from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# ROUTERS
from .routes import model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model.router, prefix="/model", tags=["model"])


@app.get("/")
async def root():
    return {"message": "Hello World from Main hello"}
