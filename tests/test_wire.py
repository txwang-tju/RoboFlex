import time
import unittest
import json

from roboflex.bridge_sim import SimulatedBridge
from roboflex.config import Config
from roboflex.planning import Command, GeometricPickPlacePolicy, Plan
from roboflex.wire import CAPABILITIES, PROTOCOL, JsonLineArm, TcpTransport


class ScriptedTransport:
    """Inject ACK/protocol faults without touching a real serial port."""
    def __init__(self, simulated=True, wrong_move_id=False, reject_stop=False):
        self.simulated, self.wrong_move_id, self.reject_stop = simulated, wrong_move_id, reject_stop
        self.q = GeometricPickPlacePolicy(Config().arm).home_joints
        self.buffer = b""
        self.sent = []
        self.closed = False

    def write(self, raw):
        request = json.loads(raw)
        self.sent.append(request)
        reply = {"id": request["id"], "status": "done", "joints_deg": self.q}
        if request["op"] == "hello":
            reply.update(protocol=PROTOCOL, capabilities=sorted(CAPABILITIES), simulated=self.simulated)
        elif request["op"] == "heartbeat":
            return
        elif request["op"] == "move_joints" and self.wrong_move_id:
            reply["id"] += 10000
        elif request["op"] == "stop" and self.reject_stop:
            reply.update(status="error", error="injected_stop_error")
        self.buffer += (json.dumps(reply) + "\n").encode()

    def read(self):
        value, self.buffer = self.buffer, b""
        return value

    def close(self):
        self.closed = True


def wait_for(arm, statuses, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if arm.poll(0) in statuses:
            return arm.poll(0)
        time.sleep(0.01)
    raise AssertionError(f"Backend remained {arm.status}: {arm.error}")


class WireTests(unittest.TestCase):
    def test_physical_handshake_refuses_unconfigured_motion(self):
        transport = ScriptedTransport(simulated=False)
        with self.assertRaises(RuntimeError):
            JsonLineArm(transport, Config())
        self.assertTrue(transport.closed)
        self.assertEqual([p["op"] for p in transport.sent], ["hello"])

    def test_wrong_ack_id_cannot_advance_plan(self):
        cfg = Config().validate()
        cfg.control.command_timeout_s = 0.1
        transport = ScriptedTransport(wrong_move_id=True)
        arm = JsonLineArm(transport, cfg)
        try:
            arm.start(Plan("test", (Command("move_joints", 0.05, arm.joints_deg), Command("gripper", 0.1, width_mm=10))), 0)
            deadline = time.monotonic() + 1
            while arm.status not in {"FAULT", "COMPLETE"} and time.monotonic() < deadline:
                arm.heartbeat(0)
                time.sleep(0.01)
            self.assertEqual(arm.status, "FAULT")
            self.assertIn("timeout", arm.error)
            self.assertNotIn("gripper", [p["op"] for p in transport.sent])
        finally:
            arm.close()

    def test_stop_rejection_faults_without_resend_loop(self):
        transport = ScriptedTransport(reject_stop=True)
        arm = JsonLineArm(transport, Config())
        try:
            arm.stop("test")
            self.assertEqual(wait_for(arm, {"FAULT"}), "FAULT")
            time.sleep(0.05)
            self.assertEqual(sum(p["op"] == "stop" for p in transport.sent), 1)
        finally:
            arm.close()

    def test_real_tcp_completion_and_measured_state(self):
        cfg = Config().validate()
        policy = GeometricPickPlacePolicy(cfg.arm)
        with SimulatedBridge(cfg) as bridge:
            arm = JsonLineArm(TcpTransport(*bridge.server_address), cfg, require_simulator=True)
            try:
                target = tuple(q + (1 if i == 0 else 0) for i, q in enumerate(policy.home_joints))
                arm.start(Plan("test", (Command("move_joints", 0.08, target), Command("gripper", 0.08, width_mm=35))), 0)
                end = time.monotonic() + 2
                while arm.poll(0) == "RUNNING" and time.monotonic() < end:
                    arm.heartbeat(0)
                    time.sleep(0.02)
                self.assertEqual(arm.status, "COMPLETE", arm.error)
                self.assertAlmostEqual(arm.joints_deg[0], target[0], places=4)
            finally:
                arm.close()

    def test_stop_preempts_waiting_for_motion_ack(self):
        cfg = Config().validate()
        with SimulatedBridge(cfg) as bridge:
            arm = JsonLineArm(TcpTransport(*bridge.server_address), cfg, require_simulator=True)
            try:
                q = tuple(v + (10 if i == 0 else 0) for i, v in enumerate(arm.joints_deg))
                arm.start(Plan("test", (Command("move_joints", 2, q), Command("gripper", 0.1, width_mm=10))), 0)
                time.sleep(0.1)
                before = time.monotonic()
                arm.stop("operator_cancel")
                self.assertEqual(wait_for(arm, {"STOPPED", "FAULT"}), "STOPPED")
                self.assertLess(time.monotonic() - before, 0.5)
                self.assertNotIn("gripper", [e["op"] for e in bridge.events])
            finally:
                arm.close()

    def test_main_loop_stall_triggers_independent_watchdog(self):
        cfg = Config().validate()
        cfg.control.watchdog_s = 0.18
        with SimulatedBridge(cfg) as bridge:
            arm = JsonLineArm(TcpTransport(*bridge.server_address), cfg, require_simulator=True)
            try:
                arm.start(Plan("test", (Command("move_joints", 2, arm.joints_deg),)), 0)
                self.assertEqual(wait_for(arm, {"FAULT"}), "FAULT")
                self.assertIn("watchdog", arm.error)
                self.assertTrue(any(e["op"] == "stop" for e in bridge.events))
            finally:
                arm.close()


if __name__ == "__main__":
    unittest.main()
