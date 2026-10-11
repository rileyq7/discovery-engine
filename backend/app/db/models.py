"""SQLModel tables. SQLite is only an index: every row is rebuilt from the files in lab/ on startup."""
from __future__ import annotations

from typing import Optional

from sqlmodel import Field, SQLModel


class Experiment(SQLModel, table=True):
    __tablename__ = "experiments"
    id: str = Field(primary_key=True)
    family: str = Field(index=True)
    hypothesis: str
    expected_if_true: str
    expected_if_false: str
    config_json: str
    state: str = Field(index=True)
    proposed_by: str
    created_at: str
    updated_at: str = ""
    verdict: Optional[str] = None
    estimate_hours: float = 0.0
    compute_hours: float = 0.0
    replicates_finding_id: Optional[str] = None
    record_json: str = "{}"  # the full lab/ record (history, notes, ...)


class Run(SQLModel, table=True):
    __tablename__ = "runs"
    id: str = Field(primary_key=True)
    experiment_id: str = Field(index=True)
    seed: int
    status: str = Field(index=True)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    compute_hours: float = 0.0
    artifact_dir: str = ""
    git_commit: Optional[str] = None
    bad: bool = False
    record_json: str = "{}"


class Metric(SQLModel, table=True):
    __tablename__ = "metrics"
    rowid: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    step: int
    name: str = Field(index=True)
    value: float


class ProbeResult(SQLModel, table=True):
    __tablename__ = "probe_results"
    rowid: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    checkpoint_step: int
    layer: int
    target: str
    r2: float
    control_r2: Optional[float] = None


class Finding(SQLModel, table=True):
    __tablename__ = "findings"
    id: str = Field(primary_key=True)
    claim: str
    verdict: str
    status: str = Field(index=True)
    evidence_run_ids: str  # JSON list
    confidence_note: str
    created_at: str
    experiment_id: Optional[str] = None
    record_json: str = "{}"


class Question(SQLModel, table=True):
    __tablename__ = "questions"
    id: str = Field(primary_key=True)
    text: str
    status: str = Field(index=True)
    linked_finding_ids: str = "[]"
    created_at: str = ""


class Paper(SQLModel, table=True):
    __tablename__ = "papers"
    id: str = Field(primary_key=True)
    title: str
    markdown: str
    cited_finding_ids: str = "[]"
    updated_at: str


class Event(SQLModel, table=True):
    __tablename__ = "events"
    id: Optional[int] = Field(default=None, primary_key=True)
    uid: str = Field(index=True)
    timestamp: str
    actor: str
    type: str
    payload_json: str
