"""Local TCP firmware simulator for integration tests; never controls an actuator."""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import time

from .config import Config
from .planning import FourAxisIK
from .types import finite
from .wire import CAPABILITIES, PROTOCOL


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        cfg = server.config
        q = list(FourAxisIK(cfg.arm).inverse(tuple(cfg.arm.home_mm)))
        width = cfg.arm.gripper_open_mm
        pending = None
        last_heartbeat = time.monotonic()
        buffer = b""
        self.request.settimeout(0.02)

        def reply(seq, status="done", **extra):
            data = {"id": seq, "status": status, "joints_deg": q, "gripper_mm": width, **extra}
            self.request.sendall((json.dumps(data, allow_nan=False) + "\n").encode())

        while not server.quit_event.is_set():
            now = time.monotonic()
            if pending:
                cmd, start, end, original_q, original_width = pending
                fraction = max(0.0, min(1.0, (now - start) / (end - start)))
                if cmd["op"] == "move_joints":
                    q = [a + (b - a) * fraction for a, b in zip(original_q, cmd["joints_deg"])]
                else:
                    width = original_width + (cmd["width_mm"] - original_width) * fraction
                if now - last_heartbeat > cfg.control.watchdog_s + 0.1:
                    server.events.append({"op": "firmware_watchdog_stop"})
                    pending = None
                elif now >= end:
                    reply(cmd["id"])
                    pending = None
            try:
                data = self.request.recv(4096)
                if not data:
                    return
                buffer += data
            except socket.timeout:
                continue
            if len(buffer) > 16384:
                return
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                cmd = json.loads(line)
                op, seq = cmd.get("op"), cmd.get("id")
                server.events.append(cmd)
                if op == "hello":
                    reply(seq, protocol=PROTOCOL, capabilities=sorted(CAPABILITIES), simulated=True)
                elif op == "heartbeat":
                    last_heartbeat = time.monotonic()
                elif op == "stop":
                    pending = None
                    reply(seq)
                elif op in {"move_joints", "gripper"}:
                    duration = cmd.get("duration_s")
                    valid = finite((duration,)) and 0.01 <= duration <= 30
                    if op == "move_joints":
                        joints = cmd.get("joints_deg", [])
                        valid &= len(joints) == 4 and finite(joints) and all(lo <= v <= hi for v, (lo, hi) in zip(joints, cfg.arm.joint_limits_deg))
                        if valid:
                            valid &= all(abs(a - b) <= cfg.arm.max_joint_speed_deg_s * duration + 0.1 for a, b in zip(q, joints))
                    else:
                        valid &= finite((cmd.get("width_mm"),)) and 0 <= cmd["width_mm"] <= cfg.arm.gripper_max_mm
                    if not valid or pending is not None:
                        reply(seq, "error", error="limits_or_busy")
                        continue
                    now = time.monotonic()
                    pending = (cmd, now, now + duration, q[:], width)
                else:
                    reply(seq, "error", error="unknown_op")


class SimulatedBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, config: Config, port: int = 0):
        self.config = config
        self.events: list[dict] = []
        self.quit_event = threading.Event()
        super().__init__(("127.0.0.1", port), _Handler)
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.quit_event.set()
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=0.2)
