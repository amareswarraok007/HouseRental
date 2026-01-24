from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class IncidentAnomaly(BaseModel):
    service: str
    metric: str
    value: float
    baseline: float
    timestamp: str


class Diagnosis(BaseModel):
    root_cause: str
    confidence: float
    evidence: List[str]


class RemediationStep(BaseModel):
    action: str
    risky: bool
    reason: str


class AgentTimelineEntry(BaseModel):
    agent: str
    status: str
    summary: str
    timestamp: str


class RunMetrics(BaseModel):
    mttr_minutes: float
    detection_minutes: float
    confidence: float


class LearningSummary(BaseModel):
    run_1: RunMetrics
    run_2: RunMetrics
    improvement: str


class IncidentResponse(BaseModel):
    incident_name: str
    stability_score: int
    anomalies: List[IncidentAnomaly]
    diagnosis: Diagnosis
    remediation: List[RemediationStep]
    execution_log: List[str]
    timeline: List[AgentTimelineEntry]
    learning: LearningSummary
    run_number: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
