"""Deterministic synthetic sequences, explicitly not project performance data."""
from dataclasses import replace

from .types import Glove, Hand, Observation, Target

SCENARIOS = ("success", "ambiguous", "no-intent", "cancel", "withdraw", "stale", "target-shift", "stale-glove")


def sample(t: float, scenario: str = "success") -> Observation:
    targets = (
        Target("0", "cup", (0.65, 0.40), (210, -70, 35), 1.0, 35),
        Target("1", "box", (0.65, 0.75), (230, 60, 35), 1.0, 30))
    tip = (0.35, 0.40)
    hand = Hand((0.20, 0.40), tip, (0.25, 0.40))
    if scenario == "ambiguous":
        targets = (replace(targets[0], uv=(0.65, 0.385)), replace(targets[1], uv=(0.65, 0.415)))
    elif scenario == "no-intent":
        hand = replace(hand, pointing=False)
    elif scenario == "withdraw" and t >= 2.0:
        dx = min(0.3, (t - 2.0) * 0.6)
        hand = Hand((0.20 - dx, 0.40), (0.35 - dx, 0.40), (0.25 - dx, 0.40))
    elif scenario == "target-shift" and t >= 3.0:
        targets = (replace(targets[0], position_mm=(260, -70, 35)), targets[1])
    glove = Glove(0.0, (0.1, 0.2, 0.1, 0.1), confirm=True) if scenario == "stale-glove" and t > 2.0 else None
    confirm = scenario not in {"ambiguous", "no-intent", "stale-glove", "withdraw"} and 2.60 <= t < 2.65
    return Observation(t, targets, hand, glove, confirm=confirm,
                       cancel=scenario == "cancel" and t >= 4.0,
                       expected_target="0" if scenario not in {"ambiguous", "no-intent"} else "__none__")
