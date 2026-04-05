"""
Веб-интерфейс к боту: тот же агент, что и в Telegram (app.agent.run_agent).

Запуск из корня репозитория:
    python -m uvicorn web.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.run_agent import clear_chat_history, run_agent, session_id_to_chat_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100)


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)


app = FastAPI(
    title="LLM Bot Web",
    description="Чат с агентом (LangGraph + RAG + web search)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chat_id = session_id_to_chat_id(req.session_id)
    logger.info("POST /api/chat  session=%s  chat_id=%s  len=%d", req.session_id, chat_id, len(req.message))
    try:
        reply = run_agent(req.message, chat_id=chat_id)
    except Exception:
        logger.exception("run_agent failed")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса") from None
    return ChatResponse(reply=reply, session_id=req.session_id)


@app.post("/api/clear")
def clear(req: ClearRequest):
    chat_id = session_id_to_chat_id(req.session_id)
    logger.info("POST /api/clear  session=%s  chat_id=%s", req.session_id, chat_id)
    clear_chat_history(chat_id)
    return {"status": "ok"}


_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
