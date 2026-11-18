"""Jalankan dari root proyek: uvicorn backend.app.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import history, model_info, predict, stats
from backend.app.core.config import get_settings
from backend.app.core.database import Base, engine
from backend.app import models  # noqa: F401  (mendaftarkan tabel)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)  # buat tabel jika belum ada
    yield


app = FastAPI(title="Arun-D", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (predict.router, history.router, stats.router, model_info.router):
    app.include_router(r)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok"}
