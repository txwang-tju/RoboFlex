from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

from .backends import MockArm
from .config import Config
from .controller import SharedAutonomyController
from .demo import SCENARIOS, sample
from .planning import GeometricPickPlacePolicy
from .types import Observation

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


class Recorder:
    def __init__(self, path: str | None, mode: str):
        self.path = Path(path) if path else Path("runs") / f"{mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.previous_state = "OBSERVING"
        self.mode = mode

    def write(self, obs, decision):
        row = {"schema": 1, "mode": self.mode, "previous_state": self.previous_state,
               "observation": obs.to_dict(), "decision": decision.to_dict()}
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.previous_state = decision.state

    def close(self):
        self.file.close()


def announce(decision, previous):
    key = (decision.state, decision.message)
    if key != previous:
        print(f"{decision.timestamp:8.2f}s | {decision.state:11s} | target={decision.target_id or '-'} | {decision.message}")
    return key


def demo(args):
    cfg = Config.load(args.config)
    policy = GeometricPickPlacePolicy(cfg.arm)
    with ExitStack() as stack:
        if args.wire:
            from .bridge_sim import SimulatedBridge
            from .wire import JsonLineArm, TcpTransport
            bridge = stack.enter_context(SimulatedBridge(cfg))
            backend = JsonLineArm(TcpTransport(*bridge.server_address), cfg, require_simulator=True)
            print("TCP firmware simulator: local loopback only, no physical robot")
        else:
            backend = MockArm(policy.home_joints)
        stack.callback(backend.close)
        recorder = Recorder(args.log, "synthetic")
        stack.callback(recorder.close)
        controller = SharedAutonomyController(cfg, backend, policy)
        previous = None
        start = time.monotonic()
        for index in range(800):
            t = round(index * 0.05, 4)
            if args.wire:
                pause = start + t - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
            obs = sample(t, args.scenario)
            # Hardware I/O watchdog uses wall time; synthetic timestamps stay deterministic.
            now = t + (1.0 if args.scenario == "stale" and t >= 3.0 else 0.0)
            decision = controller.step(obs, now)
            recorder.write(obs, decision)
            previous = announce(decision, previous)
            if decision.state in controller.TERMINAL:
                break
            if t >= 8.0 and args.scenario in {"ambiguous", "no-intent", "stale-glove"}:
                break
        print(f"Final state: {controller.state}; log: {recorder.path.resolve()}")
        if isinstance(backend, MockArm):
            print(f"Completed mock commands: {len(backend.commands)}; stop requests: {backend.stop_reasons}")


def replay(args):
    cfg = Config.load(args.config)
    policy = GeometricPickPlacePolicy(cfg.arm)
    backend = MockArm(policy.home_joints)
    recorder = Recorder(args.log, "replay")
    controller = SharedAutonomyController(cfg, backend, policy)
    previous = None
    try:
        with open(args.input, encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                raw = json.loads(line)
                obs = Observation.from_dict(raw.get("observation", raw))
                decision = controller.step(obs)
                recorder.write(obs, decision)
                previous = announce(decision, previous)
        print(f"Replay log: {recorder.path.resolve()}")
    finally:
        recorder.close()
        backend.close()


def camera(args):
    from .vision import LatestCamera, WebcamPerception, _deps
    cv2, _, _ = _deps()
    cfg = Config.load(args.config)
    policy = GeometricPickPlacePolicy(cfg.arm)
    with ExitStack() as stack:
        if args.backend == "mock":
            arm = MockArm(policy.home_joints)
            simulated = True
        else:
            from .wire import JsonLineArm, SerialTransport, TcpTransport
            if not args.allow_motion or not cfg.calibration_verified:
                raise ValueError("Physical camera mode requires --allow-motion and measured calibration_verified=true")
            if cfg.camera_calibration is None:
                raise ValueError("Physical camera mode requires camera calibration before opening the arm port")
            if args.backend == "serial":
                if not args.arm_port:
                    raise ValueError("--arm-port is required")
                transport = SerialTransport(args.arm_port)
            else:
                transport = TcpTransport(args.host, args.port)
            arm = JsonLineArm(transport, cfg, allow_motion=True)
            simulated = arm.simulated
        stack.callback(arm.close)
        perception = WebcamPerception(cfg, args.model, args.hand, simulated=simulated)
        stack.callback(perception.close)
        grabber = LatestCamera(args.camera, args.width, args.height)
        stack.callback(grabber.close)
        glove = None
        if args.glove_port:
            from .glove import SerialGlove
            glove = SerialGlove(args.glove_port)
            stack.callback(glove.close)
        recorder = Recorder(args.log, "camera")
        stack.callback(recorder.close)
        stack.callback(cv2.destroyAllWindows)
        controller = SharedAutonomyController(cfg, arm, policy)
        previous = None
        last_capture_time = None
        print("C: confirm | X: cancel | SPACE: latched stop | R: reset | Q: quit")
        print("MOCK FIXTURE COORDINATES" if simulated and not cfg.camera_calibration else "Calibrated marker pose input")
        while True:
            captured = grabber.get()
            if captured is None:
                if last_capture_time is not None and time.monotonic() - last_capture_time < cfg.control.max_observation_age_s:
                    time.sleep(0.005)
                    continue
                obs = Observation(time.monotonic())
                image = None
            else:
                timestamp, image = captured
                last_capture_time = timestamp
                obs = perception.observe(image, timestamp, glove.get() if glove else None)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            obs = replace(obs, confirm=key == ord("c"), cancel=key == ord("x"),
                          estop=key == 32, reset=key == ord("r"))
            decision = controller.step(obs, time.monotonic())
            recorder.write(obs, decision)
            previous = announce(decision, previous)
            if image is not None:
                text = f"{decision.state} target={decision.target_id} {decision.message}"
                cv2.putText(image, text[:90], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 255), 2)
                cv2.putText(image, "SIMULATION" if simulated else "PHYSICAL BRIDGE", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("RoboFlex | C confirm / X cancel / SPACE stop / R reset / Q quit", image)
            else:
                time.sleep(0.02)
        print(f"Camera log: {recorder.path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="RoboFlex intent-assisted arm prototype (mock by default)")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="Run a deterministic scenario; no camera/model needed")
    p.add_argument("--scenario", choices=SCENARIOS, default="success")
    p.add_argument("--wire", action="store_true", help="Exercise the local TCP simulator in real time")
    p.add_argument("--config")
    p.add_argument("--log")
    p.set_defaults(func=demo)
    p = commands.add_parser("replay", help="Replay JSONL observations through the mock controller")
    p.add_argument("input")
    p.add_argument("--config")
    p.add_argument("--log")
    p.set_defaults(func=replay)
    p = commands.add_parser("camera", help="Real camera input; mock arm unless explicitly configured")
    p.add_argument("--config")
    p.add_argument("--model", default="models/hand_landmarker.task")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hand", choices=("Right", "Left"), default="Right")
    p.add_argument("--backend", choices=("mock", "tcp", "serial"), default="mock")
    p.add_argument("--allow-motion", action="store_true")
    p.add_argument("--arm-port")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--glove-port")
    p.add_argument("--log")
    p.set_defaults(func=camera)
    p = commands.add_parser("download-model", help="Explicitly download the official Hand Landmarker model")
    p.add_argument("--output", default="models/hand_landmarker.task")
    def download(args):
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".downloading")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response, temp.open("wb") as out:
                while data := response.read(65536):
                    out.write(data)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        print(f"Official model: {path.resolve()}")
    p.set_defaults(func=download)
    p = commands.add_parser("markers", help="Generate ArUco IDs 0 and 1 with white printing margins")
    p.add_argument("--output", default="markers")
    def markers(args):
        from .vision import _deps
        cv2, np, _ = _deps()
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for i in range(2):
            marker = cv2.aruco.generateImageMarker(dictionary, i, 400)
            image = cv2.copyMakeBorder(marker, 70, 70, 70, 70, cv2.BORDER_CONSTANT, value=255)
            cv2.imencode(".png", image)[1].tofile(str(path / f"marker_{i}.png"))
        print(path.resolve())
    p.set_defaults(func=markers)
    p = commands.add_parser("evaluate", help="Frame/event metrics from labelled JSONL logs")
    p.add_argument("input")
    def evaluate(args):
        from .evaluation import evaluate
        print(json.dumps(evaluate(args.input), ensure_ascii=False, indent=2))
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Stopped by operator", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"RoboFlex: {exc}", file=sys.stderr)
        return 1
    return 0
