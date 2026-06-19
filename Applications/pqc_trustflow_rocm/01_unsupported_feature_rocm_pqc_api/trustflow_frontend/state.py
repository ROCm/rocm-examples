from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json
import time


@dataclass
class TrustFlowState:
    sensitive_text: str = ""
    kem_choice: str = "Kyber-768"
    sig_choice: str = "ML-DSA-65"
    batch_size: int = 1024
    mode: str = "paper"
    stage_status: dict[str, str] = field(default_factory=lambda: {
        "prepare": "idle",
        "encaps": "idle",
        "encrypt": "idle",
        "sign": "idle",
        "verify": "idle",
        "decaps": "idle",
        "decrypt": "idle",
    })
    artifacts: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    transcript: list[str] = field(default_factory=list)
    verified: bool = False

    def add_event(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.transcript.append(f"[{stamp}] {message}")

    def set_stage(self, stage: str, status: str) -> None:
        self.stage_status[stage] = status

    def set_artifact(self, name: str, value: str) -> None:
        self.artifacts[name] = value

    def set_timing(self, name: str, value: float) -> None:
        self.timings_ms[name] = value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
