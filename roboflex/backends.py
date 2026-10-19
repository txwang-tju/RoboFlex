"""Arm backends. The default deterministic mock has no network/serial side effects."""
from __future__ import annotations

from typing import Protocol

from .planning import Plan, Command


class ArmBackend(Protocol):
    joints_deg: tuple[float, ...]
    status: str
    error: str
    def start(self, plan: Plan, now: float) -> None: ...
    def poll(self, now: float) -> str: ...
    def heartbeat(self, now: float) -> None: ...
    def stop(self, reason: str) -> None: ...
    def close(self) -> None: ...


class MockArm:
    def __init__(self, home_joints: tuple[float, ...]):
        self.joints_deg = home_joints
        self.status = "IDLE"
        self.error = ""
        self.plan: Plan | None = None
        self.index = 0
        self.deadline = 0.0
        self.commands: list[Command] = []
        self.stop_reasons: list[str] = []
        self.gripper_mm = 70.0

    def start(self, plan: Plan, now: float):
        if self.status == "RUNNING":
            raise RuntimeError("Backend already executing")
        self.plan, self.index, self.status = plan, 0, "RUNNING"
        self.error = ""
        self.deadline = now + plan.commands[0].duration_s

    def heartbeat(self, now: float):
        pass

    def poll(self, now: float) -> str:
        while self.status == "RUNNING" and now >= self.deadline:
            assert self.plan is not None
            command = self.plan.commands[self.index]
            self.commands.append(command)
            if command.joints_deg is not None:
                self.joints_deg = command.joints_deg
            if command.width_mm is not None:
                self.gripper_mm = command.width_mm
            self.index += 1
            if self.index == len(self.plan.commands):
                self.status = "COMPLETE"
            else:
                self.deadline += self.plan.commands[self.index].duration_s
        return self.status

    def stop(self, reason: str):
        self.stop_reasons.append(reason)
        self.status = "STOPPED"
        self.plan = None
        # No automatic release or blind home command; keep the current gripper state.

    def close(self):
        if self.status == "RUNNING":
            self.stop("shutdown")
