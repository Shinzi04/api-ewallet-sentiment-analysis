import os
from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["model"])
async def root():
  return {"message": "Hello World"}