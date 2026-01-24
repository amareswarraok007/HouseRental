from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


DEFAULT_STATE = {
    "run_count": 0,
    "latency_multiplier": 1.6,
    "error_rate_threshold": 0.08,
    "last_confidence": 0.0,
}


class LearningStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(DEFAULT_STATE)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return DEFAULT_STATE.copy()
        return json.loads(self.path.read_text())

    def save(self, state: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(state, indent=2))

    def update_after_run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["run_count"] = state.get("run_count", 0) + 1
        if state["run_count"] >= 1:
            state["latency_multiplier"] = max(1.2, state.get("latency_multiplier", 1.6) - 0.2)
            state["error_rate_threshold"] = max(0.05, state.get("error_rate_threshold", 0.08) - 0.01)
        return state
