"""Offline frame metrics and event counts; no physical grasp success is inferred."""
from __future__ import annotations

from collections import Counter
import json

from .intent import NO_INTENT


def evaluate(path: str, threshold: float = 0.72, margin: float = 0.22) -> dict:
    counts: Counter = Counter()
    transitions: Counter = Counter()
    confusion: Counter = Counter()
    previous_state = "OBSERVING"
    last_time = None
    duration = 0.0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        obs, decision = row["observation"], row["decision"]
        state = decision["state"]
        entered_execution = state == "EXECUTING" and previous_state != "EXECUTING"
        if state != previous_state:
            transitions[state] += 1
        previous_state = state
        timestamp = obs["timestamp"]
        if last_time is not None and timestamp > last_time:
            duration += timestamp - last_time
        last_time = timestamp
        intent = decision["intent"]
        prediction = (intent["target_id"] if intent["target_id"] is not None and intent["score"] >= threshold
                      and intent["margin"] >= margin else NO_INTENT)
        expected = obs.get("expected_target")
        if expected is not None:
            counts["annotated_frames"] += 1
            counts["correct_frames"] += prediction == expected
            confusion[(expected, prediction)] += 1
            if expected == NO_INTENT and prediction != NO_INTENT:
                counts["false_positive_frames"] += 1
            if expected == NO_INTENT and entered_execution:
                counts["false_start_events"] += 1
    n = counts["annotated_frames"]
    return {
        "notice": "Frame metrics need human labels. Synthetic demos do not estimate real-world accuracy; COMPLETE means commands completed, not verified grasp.",
        "annotated_frames": n,
        "frame_accuracy": counts["correct_frames"] / n if n else None,
        "false_positive_frames": counts["false_positive_frames"],
        "false_start_events": counts["false_start_events"],
        "state_entries": dict(transitions),
        "duration_s": round(duration, 3),
        "confusion": [{"expected": a, "predicted": b, "frames": n} for (a, b), n in sorted(confusion.items())],
    }
