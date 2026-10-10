"""Emergency alert generation and dispatch manager."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from vision.disaster_detection import DetectionReport


@dataclass
class EmergencyAlert:
    """Represents a high-priority emergency broadcast payload."""

    alert_id: str
    drone_id: str
    timestamp: float
    priority: str
    persons: int
    hazards: dict[str, int]
    location: tuple[float, float]

    def to_json(self) -> str:
        """Serializes alert to JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2)

    def summary(self) -> str:
        """Returns human-readable one-line broadcast summary."""
        x, y = self.location
        loc_str = f"X={int(round(x))}, Y={int(round(y))}"
        return (
            f"ALERT | Drone: {self.drone_id} | People: {self.persons} | "
            f"Location: {loc_str} | Priority: {self.priority}"
        )


class AlertManager:
    """Manages alert rate-limiting cooldown and persistence."""

    def __init__(
        self,
        cooldown_s: float = 10.0,
        save: bool = True,
        output_dir: str = "outputs/alerts",
    ) -> None:
        self.cooldown_s = cooldown_s
        self.save = save
        self.output_dir = Path(output_dir)
        self._last_alert_time: dict[tuple[str, int], float] = {}

        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def create(self, report: DetectionReport) -> EmergencyAlert | None:
        """Creates and persists an EmergencyAlert if conditions are met.

        Returns None if priority is not HIGH or CRITICAL, or if suppressed by cooldown.
        """
        if report.priority not in ("HIGH", "CRITICAL"):
            return None

        key = (report.drone_id, report.persons)
        now = report.timestamp or time.time()
        last_time = self._last_alert_time.get(key)
        if last_time is not None and (now - last_time) < self.cooldown_s:
            return None

        self._last_alert_time[key] = now
        alert_id = f"ALERT-{uuid.uuid4().hex[:8].upper()}"

        loc = (
            report.people_positions[0]
            if report.people_positions
            else (0.0, 0.0)
        )

        alert = EmergencyAlert(
            alert_id=alert_id,
            drone_id=report.drone_id,
            timestamp=now,
            priority=report.priority,
            persons=report.persons,
            hazards=report.hazards,
            location=loc,
        )

        if self.save:
            target_path = self.output_dir / f"{alert_id}.json"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(alert.to_json())

        return alert
