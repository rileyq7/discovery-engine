from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import lab, store

router = APIRouter(tags=["findings"])


class FindingBody(BaseModel):
    claim: str
    verdict: str
    evidence_run_ids: list[str]
    confidence_note: str
    experiment_id: Optional[str] = None
    seeds_agree: Optional[bool] = None
    replicates: Optional[str] = None
    question_ids: list[str] = []
    actor: str = "human"


class StatusBody(BaseModel):
    status: Optional[str] = None
    note: Optional[str] = None
    disputed: Optional[bool] = None
    actor: str = "human"


class QuestionBody(BaseModel):
    text: Optional[str] = None
    status: Optional[str] = None
    finding_id: Optional[str] = None
    actor: str = "human"


def _enrich(f: dict) -> dict:
    return {**f, "replication_count": sum(1 for r in f.get("replications", []) if r["agrees"])}


@router.get("/findings")
def list_findings(status: Optional[str] = None):
    return [_enrich(f) for f in reversed(store.list_findings(status))]


@router.post("/findings")
def create_finding(body: FindingBody):
    data = body.model_dump(exclude={"actor"}, exclude_none=True)
    try:
        return store.create_finding(data, body.actor)
    except store.StoreError as e:
        raise HTTPException(422, str(e))


@router.get("/findings/{fid}")
def get_finding(fid: str):
    try:
        return _enrich(store.get_finding(fid))
    except KeyError:
        raise HTTPException(404, "finding not found")


@router.post("/findings/{fid}/replicate")
def replicate(fid: str, body: StatusBody = StatusBody()):
    try:
        return lab.replicate(fid, body.actor)
    except KeyError:
        raise HTTPException(404, "finding not found")
    except lab.ProposalError as e:
        raise HTTPException(422, {"errors": e.errors})


@router.post("/findings/{fid}/promote")
def promote(fid: str, body: StatusBody):
    try:
        return store.promote_finding(fid, body.status or "replicated", body.actor, body.note or "")
    except store.StoreError as e:
        raise HTTPException(422, str(e))


@router.post("/findings/{fid}/annotate")
def annotate(fid: str, body: StatusBody):
    return store.annotate_finding(fid, body.actor, body.note, body.disputed)


@router.get("/reviews")
def reviews(limit: int = 20):
    """Independent reviewer reports, newest first."""
    from agent import review

    return list(reversed(review.list_reviews()))[:limit]


@router.get("/digests")
def digests():
    """Daily highlight reels, newest first."""
    from agent import digest

    return digest.list_digests()


@router.get("/questions")
def list_questions(status: Optional[str] = None):
    return store.list_questions(status)


@router.post("/questions")
def create_question(body: QuestionBody):
    if not body.text:
        raise HTTPException(422, "text required")
    return store.create_question(body.text, body.actor)


@router.put("/questions/{qid}")
def update_question(qid: str, body: QuestionBody):
    try:
        return store.update_question(qid, body.actor, body.status, body.finding_id)
    except (store.StoreError, KeyError) as e:
        raise HTTPException(422, str(e))
