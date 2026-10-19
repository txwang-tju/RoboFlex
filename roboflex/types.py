"""Shared messages. Time is host monotonic seconds; geometry uses base-frame mm."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


def finite(values: Any) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in values)


def distance(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class Target:
    id: str
    label: str
    uv: Vec2                         # Normalized camera image coordinates, unmirrored
    position_mm: Vec3 | None = None  # Gripper centre at grasp, never inferred from RGB alone
    confidence: float = 1.0
    grip_mm: float = 25.0

    def __post_init__(self):
        if not self.id or not finite(self.uv) or len(self.uv) != 2:
            raise ValueError("Invalid target id/uv")
        if not all(0 <= v <= 1 for v in self.uv):
            raise ValueError("Target uv must be in [0, 1]")
        if self.position_mm is not None and (len(self.position_mm) != 3 or not finite(self.position_mm)):
            raise ValueError("position_mm must be three finite values")
        if not finite((self.confidence, self.grip_mm)) or not 0 <= self.confidence <= 1 or self.grip_mm < 0:
            raise ValueError("Invalid target confidence/gripper width")


@dataclass(frozen=True)
class Hand:
    wrist: Vec2
    tip: Vec2
    index_base: Vec2
    pointing: bool = True
    pinch: bool = False
    quality: float = 1.0
    landmarks: tuple[tuple[float, float, float], ...] = ()  # Optional 21 raw normalized landmarks

    def __post_init__(self):
        if any(len(v) != 2 or not finite(v) for v in (self.wrist, self.tip, self.index_base)):
            raise ValueError("Invalid hand points")
        if not finite((self.quality,)) or not 0 <= self.quality <= 1:
            raise ValueError("Invalid hand quality")
        if self.landmarks and (len(self.landmarks) != 21 or any(len(v) != 3 or not finite(v) for v in self.landmarks)):
            raise ValueError("Expected 21 finite 3D normalized landmarks")


@dataclass(frozen=True)
class Glove:
    timestamp: float
    bends: tuple[float, ...] = ()    # Per-user normalized values, 0 straight / 1 bent
    gyro_dps: Vec3 = (0.0, 0.0, 0.0)
    confirm: bool = False           # Explicit event from firmware, not arbitrary bending
    cancel: bool = False
    enabled: bool = True

    def __post_init__(self):
        if not finite((self.timestamp, *self.bends, *self.gyro_dps)) or len(self.gyro_dps) != 3:
            raise ValueError("Invalid glove sample")
        if not all(0 <= b <= 1 for b in self.bends):
            raise ValueError("bends must be normalized to [0, 1]")


@dataclass(frozen=True)
class Observation:
    timestamp: float
    targets: tuple[Target, ...] = ()
    hand: Hand | None = None
    glove: Glove | None = None
    confirm: bool = False
    cancel: bool = False
    estop: bool = False
    reset: bool = False
    expected_target: str | None = None  # Optional annotation for offline evaluation
    image_aspect: float = 1.0           # Width / height; use isotropic units for image angles

    def __post_init__(self):
        if not finite((self.timestamp,)):
            raise ValueError("Invalid timestamp")
        if not all(isinstance(v, bool) for v in (self.confirm, self.cancel, self.estop, self.reset)):
            raise ValueError("Control events must be booleans")
        if not finite((self.image_aspect,)) or self.image_aspect <= 0:
            raise ValueError("Invalid image aspect ratio")
        if len({t.id for t in self.targets}) != len(self.targets):
            raise ValueError("Duplicate target IDs")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        data = dict(data)
        data["targets"] = tuple(Target(**v) for v in data.get("targets", []))
        if data.get("hand") is not None:
            data["hand"] = Hand(**data["hand"])
        if data.get("glove") is not None:
            data["glove"] = Glove(**data["glove"])
        return cls(**data)


@dataclass(frozen=True)
class Intent:
    probabilities: dict[str, float]
    target_id: str | None
    score: float
    margin: float
    phase: str
    withdrawn: bool = False
    sensor_used: bool = False


@dataclass(frozen=True)
class Decision:
    timestamp: float
    state: str
    target_id: str | None
    message: str
    intent: Intent

    def to_dict(self) -> dict:
        return asdict(self)
