"""Research-planner boundary and optional no-render MuJoCo task regressions."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from roboflex.stack.contracts import Entity, Scene, TaskCompiler
from roboflex.stack.planner_bridge import planning_request, validate_reference


def task_scene():
    scene = Scene(100., (Entity("bottle", "bottle", (.65, -.35, .06)),),
                  (Entity("tray", "tray", (1.15, .55, .19)),))
    token = TaskCompiler().compile({"intent": "PICK_PLACE", "object_id": "bottle",
        "destination_id": "tray", "confidence": .95}, scene, now=100.,
        confirmed=True, evidence_ids=("operator:1",), task_id="test-task")
    return token, scene


def request_fixture():
    token, scene = task_scene()
    return planning_request(token, scene, (0., 0., 0.), (.1, .2, .3, .4),
        now=100., robot_model_id="reference-4r-v1", map_revision="map-1")


def reference_fixture():
    return {"schema": "roboflex.motion-reference.v1", "task_id": "test-task",
        "scene_revision": 1, "robot_model_id": "reference-4r-v1", "map_revision": "map-1",
        "frame_id": "world", "units": "m_rad_s", "valid_until_s": 105.,
        "points": [
            {"t_s": 0., "base_xy_yaw": [0., 0., 0.], "arm_rad": [.1, .2, .3, .4], "phase": "APPROACH"},
            {"t_s": 2., "base_xy_yaw": [.2, -.1, 0.], "arm_rad": [.2, .3, .4, .5], "phase": "OBSERVE"},
        ]}


class PlannerBridgeTests(unittest.TestCase):
    limits = ((-math.pi, math.pi),) * 4

    def validate(self, reference=None, request=None, now=100.):
        return validate_reference(reference or reference_fixture(), request or request_fixture(),
                                  now=now, joint_limits_rad=self.limits)

    def test_request_binds_grounded_task_robot_and_map(self):
        request = request_fixture()
        self.assertEqual(request["schema"], "roboflex.planner-request.v1")
        self.assertEqual(request["units"], "m_rad_s")
        self.assertEqual(request["robot_state"], [0., 0., 0., .1, .2, .3, .4])
        self.assertEqual([t["kind"] for t in request["task_set"]], ["GRASP", "PLACE"])
        self.assertEqual(request["task_set"][0]["target_xyz_m"], (.65, -.35, .06))
        self.assertEqual(request["task_set"][1]["depends_on"], "pick")
        self.assertEqual(request["task_set"][1]["completion"], "object_at_destination_and_released")

    def test_request_rejects_stale_scene_unbound_model_and_bad_dimensions(self):
        token, scene = task_scene()
        cases = [dict(scene=replace(scene, captured_at_s=99.)),
                 dict(base=(0, 0)), dict(arm=(0, 0, 0)),
                 dict(base=(math.nan, 0, 0)), dict(model=""), dict(map="")]
        for changes in cases:
            values = dict(scene=scene, base=(0, 0, 0), arm=(0, 0, 0, 0), model="model-1", map="map-1")
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                planning_request(token, values["scene"], values["base"], values["arm"],
                    now=100., robot_model_id=values["model"], map_revision=values["map"])

    def test_request_rejects_unverified_metric_source(self):
        token, scene = task_scene()
        scene = replace(scene, objects=(replace(scene.objects[0], pose_source="monocular_guess"),))
        with self.assertRaises(ValueError):
            planning_request(token, scene, (0, 0, 0), (0, 0, 0, 0), now=100.,
                             robot_model_id="model-1", map_revision="map-1")

    def test_reference_accepts_exact_model_units_and_increasing_times(self):
        points = self.validate()
        self.assertEqual(len(points), 2)
        self.assertEqual(points[-1]["phase"], "OBSERVE")

    def test_reference_rejects_identity_and_unit_mismatch(self):
        for key, value in (("task_id", "another-task"), ("scene_revision", 2),
                           ("scene_revision", True), ("robot_model_id", "another-robot"),
                           ("map_revision", "another-map"), ("frame_id", "camera"),
                           ("units", "mm_deg_s"), ("schema", "other.v1")):
            reference = reference_fixture()
            reference[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate(reference)

    def test_reference_rejects_expired_or_overlong_lease(self):
        for expiry in (100., 99., 111., math.nan):
            reference = reference_fixture()
            reference["valid_until_s"] = expiry
            with self.subTest(expiry=expiry), self.assertRaises(ValueError):
                self.validate(reference)
        with self.assertRaises(ValueError):
            self.validate(now=110.)  # Task token itself expired.

    def test_reference_horizon_cannot_exceed_its_lease(self):
        reference = reference_fixture()
        reference["points"][-1]["t_s"] = 6.
        with self.assertRaises(ValueError):
            self.validate(reference)

    def test_reference_rejects_unknown_commands_dimensions_or_bad_bounds(self):
        cases = [dict(t_s=-1), dict(t_s=0), dict(t_s=61), dict(t_s=math.inf),
                 dict(base_xy_yaw=[0, 0]), dict(base_xy_yaw=[11, 0, 0]),
                 dict(base_xy_yaw=[0, 0, math.nan]), dict(arm_rad=[0, 0, 0]),
                 dict(arm_rad=[0, 0, 0, 4]), dict(phase="UNSAFE_DIRECT_SERVO"),
                 dict(motor_pwm=100)]
        for mutation in cases:
            reference = reference_fixture()
            reference["points"][-1].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(reference)
        for points in ([], [reference_fixture()["points"][0]], [{}] * 10001):
            reference = reference_fixture()
            reference["points"] = points
            with self.subTest(size=len(points)), self.assertRaises(ValueError):
                self.validate(reference)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Install the sim extra for physics regression tests")
class SimulationScenarioTests(unittest.TestCase):
    """Use physical simulator state rather than queued-command acknowledgements."""
    @classmethod
    def setUpClass(cls):
        from roboflex.stack.runner import SCENARIOS, run_simulation
        cls.workspace = TemporaryDirectory(prefix="roboflex-test-")
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.reports, cls.traces = {}, {}
        for scenario in SCENARIOS:
            path = Path(cls.workspace.name) / scenario
            args = SimpleNamespace(output=path, provider="replay", gateway=None, video=False, scenario=scenario)
            with redirect_stdout(io.StringIO()):
                cls.reports[scenario] = run_simulation(args)
            cls.traces[scenario] = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]

    def test_success_requires_actual_release_and_settled_destination(self):
        report = self.reports["success"]
        self.assertEqual(report["state"], "COMPLETE")
        self.assertTrue(report["destination_verified"])
        self.assertFalse(report["attached"])
        events = report["events"]
        self.assertTrue(any(e["event"] == "grasp_attachment" for e in events))
        self.assertTrue(any(e["event"] == "release" for e in events))
        self.assertEqual(events[-1]["event"], "task_complete")
        x, y, z = self.traces["success"][-1]["object_xyz_m"]
        self.assertLess(abs(x - 1.15), .11)
        self.assertLess(abs(y - .55), .09)
        self.assertTrue(.05 < z < .15)
        self.assertIn("fixture_replay", report["provider"])

    def test_no_intent_never_starts_or_moves_base(self):
        report, trace = self.reports["no-intent"], self.traces["no-intent"]
        self.assertEqual(report["state"], "IDLE")
        self.assertFalse(report["destination_verified"])
        self.assertEqual(report["events"], [])
        self.assertTrue(all(max(abs(v) for v in p["base_xyz_yaw"]) < 1e-6 for p in trace))

    def test_cancel_while_carrying_holds_attachment_and_latches_stop(self):
        report, trace = self.reports["cancel"], self.traces["cancel"]
        self.assertEqual(report["state"], "CANCELLED")
        self.assertEqual(report["reason"], "explicit_operator_cancel")
        self.assertTrue(report["attached"])
        self.assertFalse(any(e["event"] == "release" for e in report["events"]))
        self.assertEqual(trace[-1]["wheel_rad_s"], [0., 0., 0., 0.])

    def test_network_loss_expires_local_lease_without_dropping_object(self):
        report, trace = self.reports["network-loss"], self.traces["network-loss"]
        self.assertEqual(report["state"], "CANCELLED")
        self.assertEqual(report["reason"], "control_lease_expired")
        self.assertTrue(report["attached"])
        carry = next(e["t"] for e in report["events"] if e.get("phase") == "CARRY")
        self.assertLessEqual(report["duration_s"] - carry, .42)
        self.assertEqual(trace[-1]["wheel_rad_s"], [0., 0., 0., 0.])

    def test_target_shift_causes_replan_and_can_still_finish(self):
        report = self.reports["target-shift"]
        self.assertEqual(report["state"], "COMPLETE")
        self.assertEqual(report["replans"], 1)
        self.assertTrue(report["destination_verified"])
        self.assertTrue(any(e["event"] == "replan" for e in report["events"]))

    def test_mapped_blocked_route_faults_before_motion(self):
        report, trace = self.reports["blocked"], self.traces["blocked"]
        self.assertEqual(report["state"], "FAULT")
        self.assertTrue("obstructed" in report["reason"] or "route" in report["reason"])
        self.assertEqual(trace[-1]["wheel_rad_s"], [0., 0., 0., 0.])
        self.assertFalse(report["attached"])
        self.assertTrue(all(max(abs(v) for v in p["base_xyz_yaw"]) < 1e-6 for p in trace))


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Install the sim extra for physics regression tests")
class ExecutorStateTests(unittest.TestCase):
    def setUp(self):
        from roboflex.stack.simulation import ReferenceExecutor, SimWorld
        self.world = SimWorld()
        self.addCleanup(self.world.close)
        self.executor = ReferenceExecutor(self.world)
        token, _ = task_scene()
        self.token, self.scene = token, self.world.scene(100.)

    def test_duplicate_task_is_idempotent_and_changed_content_is_rejected(self):
        self.assertTrue(self.executor.accept(self.token, self.scene, now_utc=100.))
        events = deepcopy(self.executor.events)
        self.assertFalse(self.executor.accept(self.token, self.scene, now_utc=100.))
        self.assertEqual(self.executor.events, events)
        with self.assertRaises(ValueError):
            self.executor.accept(replace(self.token, expires_at_s=109.), self.scene, now_utc=100.)

    def test_fault_requires_explicit_recovery_and_cannot_auto_restart(self):
        self.executor.accept(self.token, self.scene, now_utc=100.)
        self.executor.stop(.1, "test_fault", "FAULT")
        old_reference = self.executor.reference.copy()
        for i in range(10):
            t = .2 + i * .02
            self.executor.heartbeat(t)
            self.executor.step(t)
        self.assertEqual(self.executor.state, "FAULT")
        self.assertEqual(self.executor.reference, old_reference)
        with self.assertRaises(ValueError):
            self.executor.accept(replace(self.token, task_id="new-task"), self.scene, now_utc=100.)
        self.assertFalse(self.executor.accept(self.token, self.scene, now_utc=100.))
        self.assertEqual(self.executor.state, "FAULT")

    def test_completion_is_not_an_ack_or_finished_trajectory(self):
        self.executor.accept(self.token, self.scene, now_utc=100.)
        self.executor.phase = "VERIFY"
        self.executor.phase_since = 0.
        self.executor.heartbeat(2.)
        self.executor.step(2.)
        self.assertEqual(self.executor.state, "FAULT")
        self.assertIn("did not settle", self.executor.reason)
        self.assertFalse(self.world.inside_tray())

    def test_unverified_pose_source_is_rejected_before_execution(self):
        scene = replace(self.scene, objects=(replace(self.scene.objects[0], pose_source="monocular_guess"),))
        with self.assertRaises(ValueError):
            self.executor.accept(self.token, scene, now_utc=100.)
        self.assertEqual(self.executor.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
