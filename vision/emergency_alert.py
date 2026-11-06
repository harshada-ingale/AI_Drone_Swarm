"""Emergency alert generation and dispatch manager."""

import base64
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from vision.disaster_detection import DetectionReport
from vision.evidence import EvidenceItem


@dataclass
class EmergencyAlert:
    """Represents a high-priority emergency broadcast payload."""

    alert_id: str
    drone_id: str
    timestamp: float
    priority: str
    persons: int
    hazards: dict[str, int] | None = None
    # Deprecated: use location_gps and location_local instead
    location: tuple[float, float] = (0.0, 0.0)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    min_confidence: float | None = None
    mean_confidence: float | None = None
    needs_review: bool = False
    location_gps: list[dict[str, Any]] = field(default_factory=list)
    location_local: list[dict[str, Any]] = field(default_factory=list)
    drone: dict[str, Any] | None = None
    reference_frame: dict[str, Any] | None = None
    utc_time: str | None = None
    hazards_supported: bool = False

    def to_json(self) -> str:
        """Serializes alert to JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2)

    def summary(self) -> str:
        """Returns human-readable one-line broadcast summary."""
        if self.location_gps:
            valid_persons = [
                p
                for p in self.location_gps
                if p.get("lat") is not None and p.get("lon") is not None
            ]
            if valid_persons:
                p_strs = []
                for p in valid_persons[:3]:
                    tid = p.get("track_id", "")
                    lat = float(p["lat"])
                    lon = float(p["lon"])
                    lat_dir = "N" if lat >= 0 else "S"
                    lon_dir = "E" if lon >= 0 else "W"
                    unc = p.get("uncertainty_m")
                    unc_str = f" +/-{int(round(unc))} m" if unc is not None else ""
                    p_strs.append(
                        f"P{tid} {abs(lat):.5f}{lat_dir} {abs(lon):.5f}{lon_dir}{unc_str}"
                    )
                loc_segment = " ; ".join(p_strs)
                if len(self.location_gps) > 3:
                    loc_segment += f" ; +{len(self.location_gps) - 3} more"
            else:
                first_status = (
                    self.location_gps[0].get("location_status", "no_pose")
                    if self.location_gps
                    else "no_pose"
                )
                loc_segment = f"Loc: unknown ({first_status})"
        else:
            x, y = self.location
            loc_segment = f"Location: X={int(round(x))}, Y={int(round(y))}"

        base = (
            f"ALERT | Drone: {self.drone_id} | People: {self.persons} | "
            f"{loc_segment} | Priority: {self.priority}"
        )

        conf_part = ""
        confs = [
            e["confidence"]
            for e in self.evidence
            if isinstance(e, dict) and e.get("confidence") is not None
        ]
        if confs:
            min_c = min(confs)
            max_c = max(confs)
            if min_c == max_c:
                conf_part = f" | Conf: {min_c:.2f}"
            else:
                conf_part = f" | Conf: {min_c:.2f}-{max_c:.2f}"
        elif self.min_confidence is not None:
            if self.mean_confidence is not None and self.mean_confidence != self.min_confidence:
                conf_part = f" | Conf: {self.min_confidence:.2f}-{self.mean_confidence:.2f}"
            else:
                conf_part = f" | Conf: {self.min_confidence:.2f}"

        review_part = " | REVIEW" if self.needs_review else ""
        return f"{base}{conf_part}{review_part}"

    def to_payload(self, max_bytes: int = 24000) -> dict[str, Any]:
        """Produces a JSON-serializable dict payload for the encrypted channel.

        If payload size exceeds max_bytes, drops thumbnails one-by-one
        (most confident detections first) until within size limit.
        Never drops positions, uncertainties, or drone telemetry blocks.
        """
        payload = asdict(self)
        payload["thumbnails_dropped"] = 0

        # Check serialized size
        json_bytes = json.dumps(payload).encode("utf-8")
        if len(json_bytes) <= max_bytes:
            return payload

        # Find evidence items with thumbnails, sort by confidence DESCENDING
        ev_list = payload.get("evidence", [])
        thumb_indices = [
            i for i, item in enumerate(ev_list) if item.get("thumbnail_b64") is not None
        ]
        thumb_indices.sort(
            key=lambda idx: float(ev_list[idx].get("confidence", 0.0)),
            reverse=True,
        )

        dropped = 0
        for idx in thumb_indices:
            ev_list[idx]["thumbnail_b64"] = None
            dropped += 1
            payload["thumbnails_dropped"] = dropped
            if len(json.dumps(payload).encode("utf-8")) <= max_bytes:
                break

        return payload


class AlertManager:
    """Manages alert rate-limiting cooldown and persistence."""

    def __init__(
        self,
        cooldown_s: float = 10.0,
        save: bool = True,
        output_dir: str = "outputs/alerts",
        review_below: float = 0.50,
    ) -> None:
        self.cooldown_s = cooldown_s
        self.save = save
        self.output_dir = Path(output_dir)
        self.review_below = float(review_below)
        self._last_alert_time: dict[tuple[str, int], float] = {}

        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        report: DetectionReport,
        evidence: list[EvidenceItem] | None = None,
        location_gps: list[dict[str, Any]] | None = None,
        location_local: list[dict[str, Any]] | None = None,
        drone: dict[str, Any] | None = None,
        reference_frame: dict[str, Any] | None = None,
        utc_time: str | None = None,
        hazards_supported: bool = False,
    ) -> EmergencyAlert | None:
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

        # Process evidence
        evidence_dicts: list[dict[str, Any]] = []
        raw_thumbnails: list[bytes | None] = []
        min_conf: float | None = None
        mean_conf: float | None = None
        needs_review: bool = False

        if evidence:
            confs = []
            for item in evidence:
                if isinstance(item, dict):
                    track_id = int(item.get("track_id", 0))
                    label = str(item.get("label", "person"))
                    conf = round(float(item.get("confidence", 0.0)), 3)
                    bbox = [int(round(v)) for v in item.get("bbox", (0, 0, 0, 0))]
                    b64_str = item.get("thumbnail_b64")
                    thumb_bytes = (
                        base64.b64decode(b64_str) if b64_str is not None else None
                    )
                else:
                    track_id = int(item.track_id)
                    label = str(item.label)
                    conf = round(float(item.confidence), 3)
                    bbox = [int(round(v)) for v in item.bbox]
                    thumb_bytes = getattr(item, "thumbnail_jpeg", None)
                    b64_str = (
                        base64.b64encode(thumb_bytes).decode("ascii")
                        if thumb_bytes is not None
                        else None
                    )

                confs.append(conf)
                raw_thumbnails.append(thumb_bytes)
                evidence_dicts.append(
                    {
                        "track_id": track_id,
                        "label": label,
                        "confidence": conf,
                        "bbox": bbox,
                        "thumbnail_b64": b64_str,
                    }
                )

            if confs:
                min_conf = min(confs)
                mean_conf = round(float(sum(confs) / len(confs)), 3)
                needs_review = any(c < self.review_below for c in confs)

        # Honest hazards handling:
        # If no hazard model loaded, output null and hazards_supported=False
        rep_hazards = getattr(report, "hazards", None)
        rep_hz_supp = getattr(report, "hazards_supported", hazards_supported)
        if rep_hazards and any(v > 0 for v in rep_hazards.values()):
            actual_hazards = rep_hazards
            actual_supported = True
        elif rep_hz_supp and rep_hazards:
            actual_hazards = rep_hazards
            actual_supported = True
        else:
            actual_hazards = None
            actual_supported = False

        if utc_time is None:
            utc_time = datetime.now(timezone.utc).isoformat()

        alert = EmergencyAlert(
            alert_id=alert_id,
            drone_id=report.drone_id,
            timestamp=now,
            priority=report.priority,
            persons=report.persons,
            hazards=actual_hazards,
            location=loc,
            evidence=evidence_dicts,
            min_confidence=min_conf,
            mean_confidence=mean_conf,
            needs_review=needs_review,
            location_gps=location_gps or [],
            location_local=location_local or [],
            drone=drone,
            reference_frame=reference_frame,
            utc_time=utc_time,
            hazards_supported=actual_supported,
        )

        if self.save:
            target_path = self.output_dir / f"{alert_id}.json"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(alert.to_json())

            # Save thumbnails as outputs/alerts/alert_<id>/p<k>.jpg
            dirs_to_write = [self.output_dir / f"alert_{alert_id}"]
            if alert_id.startswith("ALERT-"):
                dirs_to_write.append(self.output_dir / f"alert_{alert_id[6:]}")

            for k, thumb_data in enumerate(raw_thumbnails):
                if thumb_data is not None:
                    for adir in dirs_to_write:
                        adir.mkdir(parents=True, exist_ok=True)
                        with open(adir / f"p{k}.jpg", "wb") as f_img:
                            f_img.write(thumb_data)

        return alert
