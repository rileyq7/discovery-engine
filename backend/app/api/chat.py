"""Lab tutor chat threads; replies stream as server-sent events."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import chat

router = APIRouter(prefix="/chat", tags=["chat"])


class NewThread(BaseModel):
    title: str = ""


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    web: bool = False  # allow WebSearch/WebFetch for this message (sends queries off the machine)
    page: Optional[str] = None  # the app page the user is looking at, e.g. "#/runs/run-…"


@router.get("/threads")
def threads():
    return chat.list_threads()


@router.post("/threads")
def new_thread(body: NewThread):
    return chat.create_thread(body.title)


@router.get("/threads/{tid}")
def thread(tid: str):
    try:
        return chat.get_thread(tid)
    except KeyError:
        raise HTTPException(404, "no such chat")


@router.delete("/threads/{tid}")
def delete(tid: str):
    chat.delete_thread(tid)
    return {"ok": True}


@router.post("/threads/{tid}/messages")
def send(tid: str, body: Message):
    try:
        chat.get_thread(tid)
    except KeyError:
        raise HTTPException(404, "no such chat")
    return StreamingResponse(chat.stream_reply(tid, body.text, body.web, body.page), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
