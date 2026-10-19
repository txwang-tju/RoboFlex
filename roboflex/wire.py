"""Non-blocking command worker for an explicit RoboFlex JSONL bridge protocol.

This is NOT an existing STM32/ESP32 protocol. Firmware must implement the contract
in docs/protocol.md, especially local watchdog, limits, stop and completion ACKs.
"""
from __future__ import annotations

import json
import socket
import threading
import time

from .config import Config
from .planning import Plan
from .types import finite

PROTOCOL = "roboflex-jsonl-v1"
CAPABILITIES = {"move_joints", "gripper", "stop", "watchdog", "state"}


class TcpTransport:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=2.0)
        self.sock.settimeout(0.03)

    def write(self, data: bytes):
        self.sock.sendall(data)

    def read(self) -> bytes:
        try:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Bridge disconnected")
            return data
        except socket.timeout:
            return b""

    def close(self):
        self.sock.close()


class SerialTransport:
    def __init__(self, port: str, baudrate: int = 115200):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Install serial extra: python -m pip install -e '.[serial]'") from exc
        # Opening some boards toggles DTR/reset. No motion is sent until handshake passes.
        self.serial = serial.Serial(port, baudrate, timeout=0, write_timeout=0.05)

    def write(self, data: bytes):
        if self.serial.write(data) != len(data):
            raise ConnectionError("Incomplete serial write")

    def read(self) -> bytes:
        return self.serial.read(min(4096, self.serial.in_waiting))

    def close(self):
        self.serial.close()


class JsonLineArm:
    def __init__(self, transport, config: Config, *, allow_motion: bool = False,
                 require_simulator: bool = False):
        self.transport, self.cfg = transport, config
        self._buffer = b""
        self._sequence = 0
        self._lock = threading.RLock()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "IDLE"
        self.error = ""
        self.joints_deg: tuple[float, ...] = ()
        self.stage = ""
        self._plan: Plan | None = None
        self._index = 0
        self._pending: tuple[int, dict, float] | None = None
        self._stop_request: str | None = None
        self._stop_fault = False
        self._last_heartbeat = time.monotonic()
        self._sent_heartbeat = 0.0
        try:
            self._send({"op": "hello", "id": 0})
            hello = None
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and hello is None:
                for packet in self._receive():
                    if packet.get("id") == 0:
                        hello = packet
                time.sleep(0.005)
            if not hello or hello.get("protocol") != PROTOCOL or not CAPABILITIES.issubset(hello.get("capabilities", [])):
                raise RuntimeError("Bridge protocol/capabilities handshake failed")
            self.simulated = hello.get("simulated") is True
            if require_simulator and not self.simulated:
                raise RuntimeError("Simulation command refuses a physical bridge")
            if not self.simulated and not (allow_motion and config.calibration_verified):
                raise RuntimeError("Physical motion needs --allow-motion and measured calibration_verified=true")
            q = hello.get("joints_deg", [])
            if len(q) != 4 or not finite(q):
                raise RuntimeError("Bridge did not supply four measured joint positions")
            self.joints_deg = tuple(q)
            self._thread = threading.Thread(target=self._run, name="roboflex-arm-io", daemon=True)
            self._thread.start()
        except Exception:
            transport.close()
            raise

    def _send(self, payload: dict):
        self.transport.write((json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))

    def _receive(self) -> list[dict]:
        self._buffer += self.transport.read()
        if len(self._buffer) > 16384:
            raise ValueError("Oversized bridge message")
        messages = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if line.strip():
                packet = json.loads(line)
                if not isinstance(packet, dict):
                    raise ValueError("Bridge packet must be an object")
                messages.append(packet)
        return messages

    def _next_id(self) -> int:
        self._sequence += 1
        return self._sequence

    def start(self, plan: Plan, now: float):
        with self._lock:
            if self.status not in {"IDLE", "COMPLETE"}:
                raise RuntimeError(f"Backend not ready: {self.status}; reconnect after a stopped task")
            if not plan.commands:
                raise RuntimeError("Empty plan")
            self._plan, self._index, self._pending = plan, 0, None
            self._stop_request, self._stop_fault = None, False
            self._last_heartbeat = time.monotonic()
            self.error, self.status = "", "RUNNING"

    def heartbeat(self, now: float):
        with self._lock:
            if self.status == "RUNNING":
                self._last_heartbeat = time.monotonic()

    def poll(self, now: float) -> str:
        with self._lock:
            return self.status

    def stop(self, reason: str):
        with self._lock:
            if self.status in {"STOPPING", "STOPPED", "FAULT"}:
                return
            self._stop_request = reason
            self.status = "STOPPING"
            self._plan = None

    def _fault(self, reason: str):
        self.error = reason
        self._stop_request = reason
        self._stop_fault = True
        self.status = "STOPPING"
        self._plan = None

    def _run(self):
        try:
            while not self._quit.is_set():
                with self._lock:
                    now = time.monotonic()
                    if self.status == "RUNNING" and now - self._last_heartbeat > self.cfg.control.watchdog_s:
                        self._fault("host_observation_watchdog_expired")
                    if self._stop_request is not None:
                        seq = self._next_id()
                        payload = {"op": "stop", "id": seq, "reason": self._stop_request}
                        self._send(payload)
                        self._pending = (seq, payload, now + self.cfg.control.command_timeout_s)
                        self._stop_request = None
                    if self.status == "RUNNING" and now - self._sent_heartbeat >= 0.10:
                        self._send({"op": "heartbeat", "id": self._next_id()})
                        self._sent_heartbeat = now
                    if self._pending and now > self._pending[2]:
                        if self._pending[1]["op"] == "stop":
                            self.error = self.error or "stop_ack_timeout"
                            self.status, self._pending = "FAULT", None
                        else:
                            self._pending = None
                            self._fault("command_completion_timeout")
                    if self.status == "RUNNING" and self._pending is None:
                        assert self._plan is not None
                        if self._index == len(self._plan.commands):
                            self.status = "COMPLETE"
                        else:
                            cmd = self._plan.commands[self._index]
                            payload = cmd.payload() | {"id": self._next_id()}
                            self.stage = cmd.stage
                            self._send(payload)
                            self._pending = (payload["id"], payload, now + cmd.duration_s + self.cfg.control.command_timeout_s)
                # Bounded read outside lock keeps stop()/heartbeat() independent of ACK wait.
                packets = self._receive()
                with self._lock:
                    for packet in packets:
                        if self._pending is None or packet.get("id") != self._pending[0]:
                            continue  # Ignore delayed ACKs for an aborted plan, never advance it.
                        command = self._pending[1]
                        if packet.get("status") != "done":
                            self._pending = None
                            reason = f"bridge_rejected: {packet.get('error', 'invalid_ack')}"
                            if command["op"] == "stop":
                                self.error, self.status = reason, "FAULT"
                            else:
                                self._fault(reason)
                            continue
                        q = packet.get("joints_deg", [])
                        if len(q) != 4 or not finite(q):
                            self._pending = None
                            if command["op"] == "stop":
                                self.error, self.status = "invalid_stop_telemetry", "FAULT"
                            else:
                                self._fault("invalid_joint_telemetry")
                            continue
                        if command["op"] == "move_joints" and any(abs(a - b) > self.cfg.arm.home_tolerance_deg for a, b in zip(q, command["joints_deg"])):
                            self._pending = None
                            self._fault("joint_tracking_error")
                            continue
                        self.joints_deg = tuple(q)
                        self._pending = None
                        if command["op"] == "stop":
                            self.status = "FAULT" if self._stop_fault else "STOPPED"
                        else:
                            self._index += 1
                time.sleep(0.005)
        except Exception as exc:
            with self._lock:
                self.error, self.status = f"bridge_io_error: {exc}", "FAULT"
                self._plan = None
                try:
                    self._send({"op": "stop", "id": self._next_id(), "reason": self.error})
                except Exception:
                    pass  # Firmware watchdog must stop even when the link is gone.

    def close(self):
        if self._thread and self._thread.is_alive():
            self.stop("shutdown")
            deadline = time.monotonic() + self.cfg.control.command_timeout_s + 0.1
            while self.poll(0) == "STOPPING" and time.monotonic() < deadline:
                time.sleep(0.01)
            self._quit.set()
            self._thread.join(timeout=0.2)
        self.transport.close()
