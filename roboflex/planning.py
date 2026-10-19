"""Reference 4R IK and a constrained tabletop pick/place baseline.

This is not whole-arm collision checking or general household motion planning.
Only configured, empty workcells and top-down compatible objects are supported.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .config import ArmConfig
from .types import Target, Vec3, distance, finite


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Command:
    op: str
    duration_s: float
    joints_deg: tuple[float, ...] | None = None
    width_mm: float | None = None
    stage: str = ""

    def payload(self) -> dict:
        data = {"op": self.op, "duration_s": round(self.duration_s, 4)}
        if self.joints_deg is not None:
            data["joints_deg"] = [round(q, 5) for q in self.joints_deg]
        if self.width_mm is not None:
            data["width_mm"] = self.width_mm
        return data


@dataclass(frozen=True)
class Plan:
    target_id: str
    commands: tuple[Command, ...]

    @property
    def duration_s(self) -> float:
        return sum(c.duration_s for c in self.commands)


class ExecutionPolicy(Protocol):
    def plan(self, target: Target, current_joints: tuple[float, ...]) -> Plan: ...


class FourAxisIK:
    def __init__(self, config: ArmConfig):
        self.cfg = config

    def inverse(self, xyz: Vec3) -> tuple[float, ...]:
        c = self.cfg
        if len(xyz) != 3 or not finite(xyz):
            raise PlanError("Nonfinite Cartesian pose")
        x, y, z = xyz
        phi = math.radians(c.tool_pitch_deg)
        rw = math.hypot(x, y) - c.tool_mm * math.cos(phi)
        zw = z - c.base_height_mm - c.tool_mm * math.sin(phi)
        cosine = (rw * rw + zw * zw - c.link1_mm**2 - c.link2_mm**2) / (2 * c.link1_mm * c.link2_mm)
        if abs(cosine) > 1 + 1e-8:
            raise PlanError(f"Unreachable pose: {xyz}")
        elbow = c.elbow_sign * math.acos(max(-1.0, min(1.0, cosine)))
        shoulder = math.atan2(zw, rw) - math.atan2(c.link2_mm * math.sin(elbow), c.link1_mm + c.link2_mm * math.cos(elbow))
        joints = tuple(math.degrees(v) for v in (math.atan2(y, x), shoulder, elbow, phi - shoulder - elbow))
        for i, (q, limits) in enumerate(zip(joints, c.joint_limits_deg)):
            if not limits[0] <= q <= limits[1]:
                raise PlanError(f"Joint {i} outside limits: {q:.2f} degrees")
        return joints

    def forward(self, joints: tuple[float, ...]) -> Vec3:
        if len(joints) != 4 or not finite(joints):
            raise PlanError("Four finite joint angles required")
        c = self.cfg
        yaw, s, e, w = (math.radians(q) for q in joints)
        r = c.link1_mm * math.cos(s) + c.link2_mm * math.cos(s + e) + c.tool_mm * math.cos(s + e + w)
        z = c.base_height_mm + c.link1_mm * math.sin(s) + c.link2_mm * math.sin(s + e) + c.tool_mm * math.sin(s + e + w)
        return (r * math.cos(yaw), r * math.sin(yaw), z)


def segment_hits_box(start: Vec3, end: Vec3, box: list) -> bool:
    """Slab intersection, including boundaries; unlike sampling it catches thin boxes."""
    low, high = box
    tmin, tmax = 0.0, 1.0
    for a, b, lo, hi in zip(start, end, low, high):
        delta = b - a
        if abs(delta) < 1e-12:
            if not lo <= a <= hi:
                return False
            continue
        t1, t2 = sorted(((lo - a) / delta, (hi - a) / delta))
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


class GeometricPickPlacePolicy:
    def __init__(self, config: ArmConfig):
        self.cfg = config
        self.ik = FourAxisIK(config)
        self.home_joints = self.ik.inverse(tuple(config.home_mm))

    def _check_point(self, xyz: Vec3):
        if not finite(xyz) or any(not lo <= v <= hi for v, lo, hi in zip(xyz, self.cfg.workspace_min_mm, self.cfg.workspace_max_mm)):
            raise PlanError(f"Pose outside configured workspace: {xyz}")

    def plan(self, target: Target, current_joints: tuple[float, ...]) -> Plan:
        c = self.cfg
        if target.position_mm is None:
            raise PlanError("No calibrated robot-base coordinates for selected target")
        if len(current_joints) != 4 or not finite(current_joints):
            raise PlanError("Missing measured joint state")
        if any(abs(a - b) > c.home_tolerance_deg for a, b in zip(current_joints, self.home_joints)):
            raise PlanError("Arm is not at the configured verified home pose; operator recovery required")
        if not 0 <= target.grip_mm <= c.gripper_open_mm <= c.gripper_max_mm:
            raise PlanError("Gripper width outside configured range")
        grasp = tuple(target.position_mm)
        place = tuple(c.handoff_mm)
        transit = max(c.transit_z_mm, grasp[2] + 50, place[2] + 50)
        current = self.ik.forward(current_joints)
        joints = current_joints
        commands: list[Command] = []

        def grip(width: float, stage: str):
            commands.append(Command("gripper", c.gripper_duration_s, width_mm=width, stage=stage))

        def move(end: Vec3, stage: str):
            nonlocal current, joints
            self._check_point(current)
            self._check_point(end)
            if any(segment_hits_box(current, end, box) for box in c.keepouts_mm):
                raise PlanError(f"End-effector segment intersects static keepout during {stage}")
            start = current
            steps = max(1, math.ceil(distance(start, end) / c.cartesian_step_mm))
            if steps + len(commands) > 500:
                raise PlanError("Trajectory has too many steps")
            for i in range(1, steps + 1):
                p = tuple(a + (b - a) * i / steps for a, b in zip(start, end))
                q = self.ik.inverse(p)
                duration = max(c.min_step_s, distance(current, p) / c.max_speed_mm_s,
                               max(abs(a - b) for a, b in zip(joints, q)) / c.max_joint_speed_deg_s)
                commands.append(Command("move_joints", duration, q, stage=stage))
                current, joints = p, q

        # Compile and validate the ENTIRE plan before sending any command.
        grip(c.gripper_open_mm, "open_before_pick")
        move((current[0], current[1], transit), "home_clearance")
        move((grasp[0], grasp[1], transit), "above_target")
        move(grasp, "descend_to_target")
        grip(target.grip_mm, "close_gripper")
        move((grasp[0], grasp[1], transit), "lift_target")
        move((place[0], place[1], transit), "above_handoff")
        move(place, "descend_to_tray")
        grip(c.gripper_open_mm, "release_on_tray")
        move((place[0], place[1], transit), "retreat")
        move((c.home_mm[0], c.home_mm[1], transit), "return_above_home")
        move(tuple(c.home_mm), "return_home")
        return Plan(target.id, tuple(commands))
