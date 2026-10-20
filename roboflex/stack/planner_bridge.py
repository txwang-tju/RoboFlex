"""Data boundary to the cooperation research planner, independent of its solver code.

This validates exchange contracts; it does not certify collision safety or implement
the original spatiotemporal optimizer / phase-weighted feedback controller.
"""
from __future__ import annotations
from dataclasses import asdict
import math
from .contracts import Scene, TaskCompiler, TaskToken, exact, number, positive_integer


def planning_request(token: TaskToken, scene: Scene, base_xy_yaw, arm_rad, *,
                     now: float, robot_model_id: str, map_revision: str) -> dict:
    obj, dst = TaskCompiler().ground(token, scene, now)
    if any(entity.pose_source not in {"simulation_truth", "calibrated_fiducial", "rgbd_registered", "verified_map"} for entity in (obj, dst)):
        raise ValueError("Metric target source must be verified before planning")
    if len(base_xy_yaw) != 3 or len(arm_rad) != 4:
        raise ValueError("Reference adapter requires planar base + four arm joints")
    state = [number(v, "robot state") for v in (*base_xy_yaw, *arm_rad)]
    if not robot_model_id or not map_revision:
        raise ValueError("A measured robot model and map revision must be bound")
    return {"schema": "roboflex.planner-request.v1", "task": token.to_dict(),
            "frame_id": "world", "units": "m_rad_s", "robot_model_id": robot_model_id,
            "map_revision": map_revision, "robot_state": state,
            "scene": asdict(scene),
            "task_set": [{"step_id": "pick", "kind": "GRASP", "entity_id": obj.id,
                          "target_xyz_m": obj.xyz_m, "completion": "object_held"},
                         {"step_id": "place", "kind": "PLACE", "entity_id": dst.id,
                          "target_xyz_m": dst.xyz_m, "depends_on": "pick",
                          "completion": "object_at_destination_and_released"}],
            "feedback_contract": "identity_revision_phase_error_completion_stop"}


def validate_reference(data: dict, request: dict, *, now: float,
                       joint_limits_rad: tuple[tuple[float, float], ...]) -> list[dict]:
    """Validate identity, age, dimensions and bounds before local geometric checks.

    Passing this function alone never authorizes motor output. The robot adapter
    must still validate the swept volume, velocities, tracking and lease locally.
    """
    exact(data, {"schema", "task_id", "scene_revision", "robot_model_id", "map_revision",
                 "frame_id", "units", "valid_until_s", "points"})
    token = TaskToken.parse(request["task"])
    token.validate_fresh(now)
    positive_integer(data.get("scene_revision"), "reference scene revision")
    expected = {"schema": "roboflex.motion-reference.v1", "task_id": token.task_id,
                "scene_revision": token.scene_revision, "robot_model_id": request["robot_model_id"],
                "map_revision": request["map_revision"], "frame_id": "world", "units": "m_rad_s"}
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError("Planner reference identity/model/map/units mismatch")
    valid_until = number(data.get("valid_until_s"), "reference expiry")
    if not now < valid_until <= token.expires_at_s:
        raise ValueError("Invalid or expired reference")
    points = data.get("points")
    if not isinstance(points, list) or not 2 <= len(points) <= 10000 or len(joint_limits_rad) != 4:
        raise ValueError("Invalid reference size or arm bounds")
    previous = -math.inf
    for point in points:
        exact(point, {"t_s", "base_xy_yaw", "arm_rad", "phase"})
        t = number(point.get("t_s"), "reference time")
        # t_s is relative to the receiving/validation time, not an absolute clock.
        if not 0 <= t <= min(60, valid_until-now) or t <= previous:
            raise ValueError("Reference times must increase within the bounded horizon")
        previous = t
        base, arm = point.get("base_xy_yaw", []), point.get("arm_rad", [])
        if len(base) != 3 or len(arm) != 4 or point.get("phase") not in {"APPROACH", "OBSERVE", "GRASP", "LIFT", "CARRY", "PLACE", "VERIFY"}:
            raise ValueError("Invalid reference dimension or phase")
        if any(abs(number(v, "base position")) > 10 for v in base[:2]):
            raise ValueError("Reference outside workspace")
        number(base[2], "yaw")
        if any(not lo <= number(v, "joint") <= hi for v, (lo, hi) in zip(arm, joint_limits_rad)):
            raise ValueError("Reference outside arm limits")
    return points
