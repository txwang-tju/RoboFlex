"""Interpretable temporal baseline, NOT a trained or calibrated intent recognizer.

Scores compete with a NO_INTENT hypothesis; timestamps, not frame counts, control
smoothing and withdrawal. Optional glove signals support phase, never spatial target ID.
"""
from __future__ import annotations

import math
from typing import Protocol

from .config import IntentConfig
from .types import Intent, Observation, distance

NO_INTENT = "__none__"


class IntentEstimator(Protocol):
    def update(self, obs: Observation) -> Intent: ...
    def reset(self) -> None: ...


class TemporalIntentEstimator:
    def __init__(self, config: IntentConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self.previous: Observation | None = None
        self.belief = {NO_INTENT: 1.0}
        self.previous_target: str | None = None
        self.withdraw_since: float | None = None

    def update(self, obs: Observation) -> Intent:
        previous = self.previous
        if previous is not None and obs.timestamp <= previous.timestamp:
            raise ValueError("Observation timestamps must strictly increase")
        dt = obs.timestamp - previous.timestamp if previous else 0.05
        # Long discontinuity starts a fresh sequence; old posture cannot carry intention.
        if dt > 0.5:
            self.belief = {NO_INTENT: 1.0}
            previous = None
            self.previous_target = None
            self.withdraw_since = None
        scores: dict[str, float] = {}
        hand = obs.hand
        sensor = obs.glove
        sensor_used = bool(sensor and sensor.enabled and 0 <= obs.timestamp - sensor.timestamp <= self.cfg.sensor_max_age_s)
        motion = (0.0, 0.0)
        if hand and previous and previous.hand:
            motion = tuple((x - y) / dt for x, y in zip(hand.tip, previous.hand.tip))
            motion = (motion[0] * obs.image_aspect, motion[1])
        phase = "no_intent"
        retracting = False
        if hand and hand.quality >= 0.5:
            direction = tuple(t - b for t, b in zip(hand.tip, hand.index_base))
            direction = (direction[0] * obs.image_aspect, direction[1])
            norm = math.hypot(*direction)
            for target in obs.targets:
                delta = tuple(c - t for c, t in zip(target.uv, hand.tip))
                delta = (delta[0] * obs.image_aspect, delta[1])
                dist = math.hypot(*delta)
                towards = sum(v * d for v, d in zip(motion, delta)) / max(dist, 1e-6)
                ray = 0.0
                if hand.pointing and norm > 1e-4 and dist > 1e-5:
                    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(direction, delta)) / (norm * dist)))
                    angle = math.acos(cosine)
                    ray = math.exp(-0.5 * (angle / self.cfg.ray_sigma_rad) ** 2) * math.exp(-dist / 2.0)
                reach = math.exp(-0.5 * (dist / 0.16) ** 2) * min(1.0, max(0.0, towards / 0.12))
                # Flexion is supporting evidence only if this user has normalized sensors.
                support = 0.0
                if sensor_used and sensor and sensor.bends and previous and previous.glove:
                    old = previous.glove
                    if len(old.bends) == len(sensor.bends) and 0 <= previous.timestamp - old.timestamp <= self.cfg.sensor_max_age_s:
                        flex_rate = sum(b - a for a, b in zip(old.bends, sensor.bends)) / (len(sensor.bends) * dt)
                        support = min(0.10, max(0.0, flex_rate) * 0.10) * max(ray, reach)
                scores[target.id] = min(1.0, max(ray, reach) + support) * target.confidence * hand.quality
                if target.id == self.previous_target and towards < -self.cfg.withdraw_speed_uv_s:
                    retracting = True
            phase = "approaching" if math.hypot(*motion) > 0.025 else "holding"
            if sensor_used and sensor and phase == "holding" and math.sqrt(sum(g*g for g in sensor.gyro_dps)) > 30:
                phase = "sensor_motion"  # Movement support, not an IMU-derived object coordinate
        else:
            scores = {t.id: 0.0 for t in obs.targets}
        if retracting:
            self.withdraw_since = self.withdraw_since if self.withdraw_since is not None else obs.timestamp
        else:
            self.withdraw_since = None
        withdrawn = self.withdraw_since is not None and obs.timestamp - self.withdraw_since >= self.cfg.withdraw_hold_s
        if retracting:
            phase = "withdrawing"
        strongest = max(scores.values(), default=0.0)
        raw = {key: value for key, value in scores.items()}
        raw[NO_INTENT] = 0.08 + 1.0 * (1.0 - strongest)
        total = sum(raw.values())
        likelihood = {k: v / total for k, v in raw.items()}
        alpha = 1.0 - math.exp(-dt / self.cfg.smoothing_s)
        # Disappeared IDs transfer their probability to NO_INTENT, never to another object.
        lost = sum(v for k, v in self.belief.items() if k not in raw)
        prior = {k: self.belief.get(k, 0.0) + (lost if k == NO_INTENT else 0.0) for k in raw}
        self.belief = {k: (1 - alpha) * prior[k] + alpha * likelihood[k] for k in raw}
        ranked = sorted(self.belief.items(), key=lambda item: item[1], reverse=True)
        key, value = ranked[0]
        margin = value - (ranked[1][1] if len(ranked) > 1 else 0.0)
        selected = key if key != NO_INTENT else None
        if selected is None:
            phase = "no_intent" if not retracting else phase
        if selected and value >= 0.6:
            self.previous_target = selected
        elif selected is None and not retracting:
            self.previous_target = None
        self.previous = obs
        return Intent(dict(self.belief), selected, value, margin, phase, withdrawn, sensor_used)
