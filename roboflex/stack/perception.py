"""Bounded streams, sensor features and event-triggered semantic windows."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import threading
from .contracts import identifier, number


def hand_features(points: list) -> dict:
    """MediaPipe image landmarks -> numeric features, never robot-world coordinates."""
    if len(points) != 21 or any(len(p) < 3 for p in points):
        raise ValueError("21 MediaPipe xyz landmarks required")
    pts = [tuple(number(v, "landmark") for v in p[:3]) for p in points]
    bends = []
    for a, b, c in ((1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        u = tuple(pts[a][i] - pts[b][i] for i in range(3))
        v = tuple(pts[c][i] - pts[b][i] for i in range(3))
        norm = math.sqrt(sum(x*x for x in u) * sum(x*x for x in v))
        if norm < 1e-9:
            raise ValueError("Degenerate hand geometry")
        angle = math.acos(max(-1, min(1, sum(x*y for x, y in zip(u, v)) / norm)))
        bends.append(round(1 - angle / math.pi, 4))
    return {"source": "mediapipe_image_landmarks", "bends_proxy": bends,
            "wrist_image_xyz": pts[0], "index_tip_image_xyz": pts[8],
            "coordinate_note": "image-normalized x/y and relative z, not metric world pose"}


def glove_features(data: dict) -> dict:
    bends = data.get("bends", [])
    gyro = data.get("gyro_dps", [])
    if not 1 <= len(bends) <= 10 or len(gyro) != 3:
        raise ValueError("Glove needs calibrated flex values and gyro_dps xyz")
    bends = [number(v, "bend") for v in bends]
    gyro = [number(v, "gyro") for v in gyro]
    if any(not 0 <= v <= 1 for v in bends) or any(abs(v) > 4000 for v in gyro):
        raise ValueError("Glove values outside configured reference range")
    return {"source": "flex_imu", "bends": bends, "gyro_dps": gyro,
            "motion_dps": round(math.sqrt(sum(v*v for v in gyro)), 3)}


@dataclass(frozen=True)
class Packet:
    stream: str
    session: str
    seq: int
    captured_at_s: float
    payload: dict

    def __post_init__(self):
        identifier(self.stream); identifier(self.session)
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise ValueError("Invalid sequence")
        number(self.captured_at_s, "capture time")


class LatestStreams:
    """One slot per known stream. Start a new receiver for a changed device session."""
    def __init__(self, allowed=("operator", "robot", "glove")):
        self.allowed = set(allowed)
        self.slots: dict[str, Packet] = {}
        self.replaced = 0
        self.lock = threading.Lock()

    def put(self, packet: Packet):
        if packet.stream not in self.allowed:
            raise ValueError("Unknown stream")
        with self.lock:
            previous = self.slots.get(packet.stream)
            if previous and (packet.session != previous.session or packet.seq <= previous.seq
                             or packet.captured_at_s < previous.captured_at_s):
                raise ValueError("Out-of-order packet or changed session; reconnect explicitly")
            self.replaced += previous is not None
            self.slots[packet.stream] = packet

    def window(self, now: float, max_age=.75, max_skew=.15) -> dict:
        with self.lock:
            if set(self.slots) != self.allowed:
                raise ValueError("Waiting for all input streams")
            packets = tuple(self.slots.values())
        if any(not -.05 <= now - p.captured_at_s <= max_age for p in packets):
            raise ValueError("Stale stream or unsynchronized clock")
        if max(p.captured_at_s for p in packets) - min(p.captured_at_s for p in packets) > max_skew:
            raise ValueError("Sensor/camera window is not aligned")
        return {p.stream: {"id": f"{p.session}:{p.seq}", "captured_at_s": p.captured_at_s,
                           **p.payload} for p in packets}


class EventTrigger:
    def __init__(self, cooldown_s=.75):
        self.cooldown_s = cooldown_s
        self.last_time = -math.inf
        self.last_signature = None

    def ready(self, phase: str, candidates: tuple[str, ...], now: float, *, cancel=False):
        signature = (phase, tuple(sorted(candidates)))
        if cancel:
            return True
        if signature == self.last_signature or now - self.last_time < self.cooldown_s:
            return False
        self.last_signature, self.last_time = signature, now
        return phase in {"holding", "pointing", "withdrawing", "changed"}


def structured_hand_text(visual: dict, glove: dict) -> str:
    """Deterministic serialization preserves source/units instead of inventing narrative."""
    return json.dumps({"visual_hand": visual, "glove": glove_features(glove)},
                      ensure_ascii=False, sort_keys=True, allow_nan=False)
