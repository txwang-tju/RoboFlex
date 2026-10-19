from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class IntentConfig:
    smoothing_s: float = 0.22
    select_score: float = 0.72
    select_margin: float = 0.22
    dwell_s: float = 0.65
    ray_sigma_rad: float = 0.20
    sensor_max_age_s: float = 0.15
    withdraw_speed_uv_s: float = 0.25
    withdraw_hold_s: float = 0.16
    pinch_hold_s: float = 0.35


@dataclass
class ControlConfig:
    max_observation_age_s: float = 0.30
    hand_loss_timeout_s: float = 0.65
    confirmation_timeout_s: float = 5.0
    max_target_shift_mm: float = 15.0
    max_uv_shift: float = 0.12
    command_timeout_s: float = 1.0
    watchdog_s: float = 0.45


@dataclass
class ArmConfig:
    # Reference geometry: base yaw + shoulder + elbow + wrist pitch. NOT your measured arm.
    link1_mm: float = 160.0
    link2_mm: float = 160.0
    tool_mm: float = 40.0
    base_height_mm: float = 60.0
    tool_pitch_deg: float = -90.0
    elbow_sign: int = -1
    joint_limits_deg: list[list[float]] = field(default_factory=lambda: [
        [-140, 140], [-10, 170], [-165, 0], [-180, 80]])
    workspace_min_mm: list[float] = field(default_factory=lambda: [70, -220, 10])
    workspace_max_mm: list[float] = field(default_factory=lambda: [330, 220, 260])
    home_mm: list[float] = field(default_factory=lambda: [170, 0, 180])
    handoff_mm: list[float] = field(default_factory=lambda: [180, 100, 35])
    transit_z_mm: float = 150.0
    cartesian_step_mm: float = 10.0
    max_speed_mm_s: float = 100.0
    max_joint_speed_deg_s: float = 35.0
    min_step_s: float = 0.05
    gripper_open_mm: float = 70.0
    gripper_max_mm: float = 80.0
    gripper_duration_s: float = 0.4
    home_tolerance_deg: float = 3.0
    # Static exclusion boxes [[xmin,ymin,zmin], [xmax,ymax,zmax]]. Endpoint path checks only.
    keepouts_mm: list = field(default_factory=list)


@dataclass
class Config:
    intent: IntentConfig = field(default_factory=IntentConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    calibration_verified: bool = False
    camera_calibration: dict | None = None
    targets: dict = field(default_factory=lambda: {
        "0": {"label": "cup", "position_mm": [210, -70, 35], "grip_mm": 35},
        "1": {"label": "box", "position_mm": [230, 60, 35], "grip_mm": 30},
    })

    def validate(self) -> "Config":
        if not isinstance(self.calibration_verified, bool):
            raise ValueError("calibration_verified must be a JSON boolean")
        if not isinstance(self.targets, dict) or any(not isinstance(k, str) or not isinstance(v, dict) for k, v in self.targets.items()):
            raise ValueError("targets must map string IDs to configurations")
        for section in (self.intent, self.control):
            for name, value in asdict(section).items():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid configuration: {name}")
        if not 0 < self.intent.select_score < 1 or not 0 < self.intent.select_margin < 1:
            raise ValueError("Selection score/margin must be between 0 and 1")
        a = self.arm
        for name in ("link1_mm", "link2_mm", "tool_mm", "cartesian_step_mm", "max_speed_mm_s",
                     "max_joint_speed_deg_s", "min_step_s", "gripper_duration_s", "home_tolerance_deg"):
            if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
                raise ValueError(f"Invalid arm parameter: {name}")
        if a.elbow_sign not in (-1, 1) or len(a.joint_limits_deg) != 4:
            raise ValueError("Reference IK requires four joints and elbow_sign -1 or 1")
        for limits in a.joint_limits_deg:
            if len(limits) != 2 or not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]:
                raise ValueError("Invalid joint limits")
        for xyz in (a.workspace_min_mm, a.workspace_max_mm, a.home_mm, a.handoff_mm):
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                raise ValueError("Invalid XYZ configuration")
        if any(lo >= hi for lo, hi in zip(a.workspace_min_mm, a.workspace_max_mm)):
            raise ValueError("Invalid workspace")
        if not all(math.isfinite(v) for v in (a.base_height_mm, a.tool_pitch_deg, a.transit_z_mm,
                                              a.gripper_open_mm, a.gripper_max_mm)):
            raise ValueError("Nonfinite arm parameter")
        if not 0 <= a.gripper_open_mm <= a.gripper_max_mm:
            raise ValueError("Invalid gripper range")
        if not a.workspace_min_mm[2] <= a.transit_z_mm <= a.workspace_max_mm[2]:
            raise ValueError("Transit height outside workspace")
        for box in a.keepouts_mm:
            if len(box) != 2 or any(len(p) != 3 for p in box):
                raise ValueError("Invalid keepout box")
            if not all(math.isfinite(v) for p in box for v in p) or any(x >= y for x, y in zip(*box)):
                raise ValueError("Invalid keepout bounds")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            return cls().validate()
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        d["intent"] = IntentConfig(**d.get("intent", {}))
        d["control"] = ControlConfig(**d.get("control", {}))
        d["arm"] = ArmConfig(**d.get("arm", {}))
        return cls(**d).validate()
