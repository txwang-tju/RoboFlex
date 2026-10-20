"""Versioned contracts. SI units and UTC seconds across hosts; local leases use monotonic time."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name}: finite number required")
    return float(value)


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
        raise ValueError("Invalid identifier")
    return value


def positive_integer(value: Any, name: str):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}: positive integer required")


def exact(data: dict, allowed: set[str]):
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Unexpected contract fields")


@dataclass(frozen=True)
class Entity:
    id: str
    label: str
    xyz_m: tuple[float, float, float] | None
    frame_id: str = "world"
    pose_source: str = "simulation_truth"
    version: int = 1

    def __post_init__(self):
        identifier(self.id)
        positive_integer(self.version, "entity version")
        if self.xyz_m is not None:
            if len(self.xyz_m) != 3 or any(abs(number(v, "position")) > 10 for v in self.xyz_m):
                raise ValueError("Position outside reference workspace")
        if self.frame_id != "world":
            raise ValueError("Grounding requires a world-frame entity version")

    @classmethod
    def parse(cls, data: dict):
        exact(data, {"id", "label", "xyz_m", "frame_id", "pose_source", "version"})
        d = dict(data)
        if d.get("xyz_m") is not None:
            d["xyz_m"] = tuple(d["xyz_m"])
        return cls(**d)


@dataclass(frozen=True)
class Scene:
    captured_at_s: float
    objects: tuple[Entity, ...]
    destinations: tuple[Entity, ...]
    revision: int = 1
    clock_uncertainty_s: float = 0.0

    def __post_init__(self):
        number(self.captured_at_s, "scene timestamp")
        positive_integer(self.revision, "scene revision")
        if not 0 <= number(self.clock_uncertainty_s, "clock uncertainty") <= .1:
            raise ValueError("Synchronize clocks before spatial execution")
        ids = [v.id for v in (*self.objects, *self.destinations)]
        if len(ids) != len(set(ids)) or self.revision < 1:
            raise ValueError("Duplicate IDs or invalid scene revision")

    @classmethod
    def parse(cls, data: dict):
        exact(data, {"captured_at_s", "objects", "destinations", "revision", "clock_uncertainty_s"})
        d = dict(data)
        for name in ("objects", "destinations"):
            d[name] = tuple(Entity.parse(v) for v in d.get(name, []))
        return cls(**d)


@dataclass(frozen=True)
class TaskToken:
    task_id: str
    object_id: str
    destination_id: str
    scene_revision: int
    object_version: int
    issued_at_s: float
    expires_at_s: float
    evidence_ids: tuple[str, ...]
    schema: str = "roboflex.task.v1"
    op: str = "PICK_PLACE"

    def __post_init__(self):
        positive_integer(self.scene_revision, "scene revision")
        positive_integer(self.object_version, "object version")
        for value in (self.task_id, self.object_id, self.destination_id):
            identifier(value)
        if self.schema != "roboflex.task.v1" or self.op != "PICK_PLACE":
            raise ValueError("Unsupported task schema/action")
        lifetime = number(self.expires_at_s, "expiry") - number(self.issued_at_s, "issued")
        if not 0 < lifetime <= 60 or self.scene_revision < 1 or self.object_version < 1:
            raise ValueError("Invalid task lifetime or versions")
        if not self.evidence_ids or len(self.evidence_ids) > 16:
            raise ValueError("Task needs bounded evidence references")
        for value in self.evidence_ids:
            identifier(value)

    def validate_fresh(self, now: float):
        if not self.issued_at_s - .1 <= now < self.expires_at_s:
            raise ValueError("Expired task or clock mismatch")

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    @classmethod
    def parse(cls, data: dict):
        exact(data, set(cls.__dataclass_fields__))
        d = dict(data)
        d["evidence_ids"] = tuple(d.get("evidence_ids", []))
        return cls(**d)


class TaskCompiler:
    """The VLM selects IDs. Geometry comes only from a fresh scene provider."""
    def compile(self, proposal: dict, scene: Scene, *, now: float, confirmed: bool,
                evidence_ids: tuple[str, ...], task_id: str) -> TaskToken | None:
        if type(confirmed) is not bool:
            raise ValueError("Confirmation must be a boolean")
        exact(proposal, {"intent", "object_id", "destination_id", "confidence", "reason"})
        intent = proposal.get("intent")
        if intent not in {"NO_INTENT", "CLARIFY", "CANCEL", "PICK_PLACE"}:
            raise ValueError("Unknown semantic intent")
        confidence = number(proposal.get("confidence", 0), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence outside [0,1]")
        if intent != "PICK_PLACE" or not confirmed or confidence < .8:
            return None
        age = now - scene.captured_at_s
        if not -scene.clock_uncertainty_s <= age <= .75 - scene.clock_uncertainty_s:
            raise ValueError("Stale spatial grounding")
        objects = {v.id: v for v in scene.objects}
        destinations = {v.id: v for v in scene.destinations}
        obj = objects.get(proposal.get("object_id"))
        dst = destinations.get(proposal.get("destination_id"))
        if obj is None or dst is None or obj.xyz_m is None or dst.xyz_m is None:
            raise ValueError("Task refers to an unknown or ungrounded entity")
        if obj.id == dst.id:
            raise ValueError("Object and destination must differ")
        return TaskToken(task_id, obj.id, dst.id, scene.revision, obj.version, now, now + 10, evidence_ids)

    def ground(self, token: TaskToken, scene: Scene, now: float):
        token.validate_fresh(now)
        if token.scene_revision != scene.revision or not -scene.clock_uncertainty_s <= now - scene.captured_at_s <= .75-scene.clock_uncertainty_s:
            raise ValueError("Scene changed or became stale before dispatch")
        obj = next((v for v in scene.objects if v.id == token.object_id), None)
        dst = next((v for v in scene.destinations if v.id == token.destination_id), None)
        if not obj or not dst or obj.version != token.object_version or obj.xyz_m is None or dst.xyz_m is None:
            raise ValueError("Grounding changed before dispatch")
        return obj, dst
