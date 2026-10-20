"""Contract, freshness and gateway regression tests for the cloud/edge stack.

These tests use deterministic intent fixtures and fake camera clients. They make
no external model request and do not require cameras, motors or MuJoCo.
"""
from dataclasses import asdict, replace
import json
import math
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from roboflex.stack.contracts import Entity, Scene, TaskCompiler, TaskToken, number
from roboflex.stack.motion import GridPlanner, Mecanum, slew
from roboflex.stack.perception import (
    EventTrigger, LatestStreams, Packet, glove_features, hand_features,
    structured_hand_text,
)
from roboflex.stack.service import IntentService, make_server
from roboflex.stack.streams import CameraStreamSource
from roboflex.stack.vlm import CompatibleVLM, ReplayVLM, validate_window


def scene_at(now=100.0):
    return Scene(now, (Entity("bottle", "bottle", (.65, -.35, .06)),),
                 (Entity("tray", "tray", (1.15, .55, .10)),))


def window_at(now=100.0, confirmed=True):
    return {"captured_at_s": now, "confirmed": confirmed,
            "evidence_ids": ["operator:1", "glove:1", "robot:1"],
            "hand_text": "{\"phase\":\"pointing\"}", "images": []}


def payload_at(now=100.0, confirmed=True):
    return {"window": window_at(now, confirmed), "scene": asdict(scene_at(now))}


def proposal(**changes):
    result = ReplayVLM().infer({}, {})
    result.update(changes)
    return result


class ContractsTests(unittest.TestCase):
    def compile(self, p=None, scene=None, **changes):
        options = dict(now=100.0, confirmed=True, evidence_ids=("operator:1",), task_id="task-1")
        options.update(changes)
        return TaskCompiler().compile(p or proposal(), scene or scene_at(), **options)

    def test_token_roundtrip_and_digest(self):
        token = self.compile()
        decoded = TaskToken.parse(json.loads(json.dumps(token.to_dict())))
        self.assertEqual(token, decoded)
        self.assertEqual(token.digest, decoded.digest)
        self.assertNotEqual(token.digest, replace(token, task_id="task-2").digest)
        obj, dst = TaskCompiler().ground(decoded, scene_at(), 100.1)
        self.assertEqual((obj.id, dst.id), ("bottle", "tray"))

    def test_only_high_confidence_confirmed_pick_can_be_compiled(self):
        for intent in ("NO_INTENT", "CLARIFY", "CANCEL"):
            with self.subTest(intent=intent):
                self.assertIsNone(self.compile(proposal(intent=intent)))
        self.assertIsNone(self.compile(confirmed=False))
        self.assertIsNone(self.compile(proposal(confidence=.799)))
        self.assertIsNotNone(self.compile(proposal(confidence=.8)))

    def test_model_cannot_invent_coordinates_or_motor_commands(self):
        for extra in ({"xyz_m": [0, 0, 0]}, {"joint_angles": [0]}, {"force_n": 5}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.compile(proposal(**extra))

    def test_unknown_or_ungrounded_entity_rejected(self):
        for p in (proposal(object_id="imaginary"), proposal(destination_id="imaginary")):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.compile(p)
        ungrounded = replace(scene_at(), objects=(Entity("bottle", "bottle", None),))
        with self.assertRaises(ValueError):
            self.compile(scene=ungrounded)

    def test_stale_future_and_clock_uncertainty_rejected(self):
        for old in (99.0, 100.01):
            with self.subTest(timestamp=old), self.assertRaises(ValueError):
                self.compile(scene=replace(scene_at(), captured_at_s=old))
        # A 700 ms old scene is outside a 750 ms budget with 100 ms uncertainty.
        with self.assertRaises(ValueError):
            self.compile(scene=replace(scene_at(), captured_at_s=99.3, clock_uncertainty_s=.1))

    def test_changed_revision_or_object_version_rejected_at_dispatch(self):
        token = self.compile()
        changed_object = replace(scene_at(), objects=(replace(scene_at().objects[0], version=2),))
        for changed in (replace(scene_at(), revision=2), changed_object,
                        replace(scene_at(), captured_at_s=99.0)):
            with self.subTest(scene=changed), self.assertRaises(ValueError):
                TaskCompiler().ground(token, changed, 100.0)

    def test_expiry_and_future_token_rejected(self):
        token = self.compile()
        for now in (99.0, 110.0, 111.0):
            with self.subTest(now=now), self.assertRaises(ValueError):
                token.validate_fresh(now)
        token.validate_fresh(109.99)

    def test_contracts_reject_unknown_fields_and_bad_schema(self):
        token_data = self.compile().to_dict()
        for mutation in ({"schema": "other.v1"}, {"op": "RUN_ARBITRARY_CODE"},
                         {"expires_at_s": 200}, {"evidence_ids": []}, {"task_id": "../bad"}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                TaskToken.parse(dict(token_data, **mutation))
        with self.assertRaises(ValueError):
            TaskToken.parse(dict(token_data, motor_command="unsafe"))
        with self.assertRaises(ValueError):
            Entity("bad", "bad", (0, 0, 0), frame_id="image")

    def test_no_bool_nan_or_infinity_numbers(self):
        for value in (True, math.nan, math.inf, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value, "test")

    def test_versions_and_confirmation_have_strict_types(self):
        for value in (True, 1.5, "1", 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Entity("bottle", "bottle", (0, 0, 0), version=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(scene_at(), revision=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.compile(), object_version=value)
        for value in (1, "yes", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compile(confirmed=value)

    def test_dispatch_accounts_for_clock_uncertainty(self):
        token = self.compile()
        ambiguous_age = replace(scene_at(), captured_at_s=99.3, clock_uncertainty_s=.1)
        with self.assertRaises(ValueError):
            TaskCompiler().ground(token, ambiguous_age, 100)

    def test_scene_ids_are_unique_across_objects_and_destinations(self):
        with self.assertRaises(ValueError):
            Scene(100, (Entity("same", "obj", (0, 0, 0)),),
                  (Entity("same", "dst", (1, 0, 0)),))


class PerceptionTests(unittest.TestCase):
    def packet(self, stream, seq=1, captured=100.0, session="session"):
        return Packet(stream, session, seq, captured, {"sample": seq})

    def streams(self):
        streams = LatestStreams()
        for stream in ("operator", "robot", "glove"):
            streams.put(self.packet(stream))
        return streams

    def test_latest_frame_has_one_slot_and_preserves_source_ids(self):
        streams = self.streams()
        for seq in range(2, 20):
            streams.put(self.packet("robot", seq, 100.05))
        result = streams.window(100.1)
        self.assertEqual(len(streams.slots), 3)
        self.assertEqual(streams.replaced, 18)
        self.assertEqual(result["robot"]["sample"], 19)
        self.assertEqual(result["robot"]["id"], "session:19")

    def test_rejects_out_of_order_changed_session_and_unknown_stream(self):
        streams = self.streams()
        cases = (self.packet("robot", 1), self.packet("robot", 0),
                 self.packet("robot", 2, 99.9), self.packet("robot", 2, session="new"),
                 self.packet("unknown"))
        for packet in cases:
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                streams.put(packet)

    def test_missing_stale_and_skewed_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            LatestStreams().window(100)
        with self.assertRaises(ValueError):
            self.streams().window(101)
        streams = self.streams()
        streams.put(self.packet("robot", 2, 100.2))
        with self.assertRaises(ValueError):
            streams.window(100.2)

    def test_sequence_must_be_nonnegative_integer(self):
        for value in (True, -1, .5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.packet("robot", seq=value)

    def test_event_trigger_deduplicates_and_cancel_bypasses_cooldown(self):
        trigger = EventTrigger(cooldown_s=.75)
        self.assertTrue(trigger.ready("pointing", ("a", "b"), 1.0))
        self.assertFalse(trigger.ready("pointing", ("b", "a"), 2.0))
        self.assertFalse(trigger.ready("holding", ("a",), 1.1))
        self.assertTrue(trigger.ready("holding", ("a",), 2.0))
        self.assertTrue(trigger.ready("holding", ("a",), 2.01, cancel=True))

    def test_sensor_features_keep_units_and_validate_ranges(self):
        data = {"bends": [.1, .5, .8], "gyro_dps": [3, 4, 0]}
        features = glove_features(data)
        self.assertEqual(features["motion_dps"], 5)
        description = json.loads(structured_hand_text({"source": "visual"}, data))
        self.assertEqual(description["glove"]["source"], "flex_imu")
        for invalid in ({"bends": [1.1], "gyro_dps": [0, 0, 0]},
                        {"bends": [.5], "gyro_dps": [0, 0]},
                        {"bends": [.5], "gyro_dps": [0, 0, math.nan]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                glove_features(invalid)

    def test_image_landmarks_are_features_not_metric_robot_pose(self):
        points = [[i / 25, (i % 5) / 6, (i % 3) / 20] for i in range(21)]
        features = hand_features(points)
        self.assertEqual(len(features["bends_proxy"]), 5)
        self.assertTrue(all(0 <= v <= 1 for v in features["bends_proxy"]))
        self.assertIn("not metric world pose", features["coordinate_note"])
        with self.assertRaises(ValueError):
            hand_features([[0, 0, 0]] * 21)
        with self.assertRaises(ValueError):
            hand_features(points[:20])


class VLMTests(unittest.TestCase):
    def test_window_requires_boolean_confirmation_and_fresh_inputs(self):
        for confirmed in ("yes", 1, None):
            with self.subTest(confirmed=confirmed), self.assertRaises(ValueError):
                validate_window(window_at(confirmed=confirmed), 100)
        with self.assertRaises(ValueError):
            validate_window(window_at(99), 100)

    def test_window_rejects_unbounded_or_malformed_images(self):
        for images in ([{"view": "robot", "data_url": "https://example.com/a.jpg"}],
                       [{"view": "robot", "data_url": "data:image/png;base64,!!!!"}],
                       [{}] * 9):
            with self.subTest(images=images), self.assertRaises(ValueError):
                validate_window(dict(window_at(), images=images), 100)

    def test_replay_provider_is_explicitly_labelled(self):
        self.assertIn("NOT_live", ReplayVLM.name)

    def test_live_provider_requires_config_and_secure_remote_url(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            CompatibleVLM()
        for base in ("http://remote.example/v1", "https://user:secret@remote.example/v1",
                     "https://remote.example/v1?secret=bad"):
            with self.subTest(base=base), patch.dict("os.environ", {
                "ROBOFLEX_VLM_BASE_URL": base, "ROBOFLEX_VLM_MODEL": "configured-model",
                "ROBOFLEX_VLM_API_KEY": "unit-test-not-a-real-key",
            }, clear=True), self.assertRaises(ValueError):
                CompatibleVLM()


class ServiceTests(unittest.TestCase):
    def test_fresh_confirmed_proposal_becomes_task(self):
        service = IntentService(ReplayVLM())
        result = service.infer(payload_at(), now=100)
        self.assertEqual(result["status"], "TASK_READY")
        token = TaskToken.parse(result["task"])
        self.assertEqual(token.evidence_ids, ("operator:1", "glove:1", "robot:1"))
        self.assertEqual(len(service.pending), 0)

    def test_confirmation_can_arrive_after_inference(self):
        service = IntentService(ReplayVLM())
        result = service.infer(payload_at(confirmed=False), now=100)
        self.assertIsNone(result["task"])
        follow_up = dict(payload_at(100.2), proposal_id=result["proposal_id"])
        committed = service.commit(follow_up, now=100.2)
        self.assertEqual(committed["status"], "TASK_READY")
        with self.assertRaises(ValueError):
            service.commit(follow_up, now=100.2)  # Already consumed.

    def test_slow_model_needs_fresh_confirmation_not_refreshed_old_timestamps(self):
        class SlowFixture:
            name = "slow_test_fixture"
            def infer(self, *_):
                return proposal()
        service = IntentService(SlowFixture())
        with patch("roboflex.stack.service.time.time", return_value=101.0):
            result = service.infer(payload_at(100), now=100)
        self.assertEqual(result["status"], "AWAITING_FRESH_CONFIRMATION")
        self.assertIsNone(result["task"])
        with self.assertRaises(ValueError):
            service.commit(dict(payload_at(100), proposal_id=result["proposal_id"]), now=101)
        fresh = service.commit(dict(payload_at(101.1), proposal_id=result["proposal_id"]), now=101.1)
        self.assertEqual(fresh["status"], "TASK_READY")
        self.assertEqual(fresh["task"]["issued_at_s"], 101.1)

    def test_cancel_invalidates_in_flight_inference_and_busy_request(self):
        entered, release = threading.Event(), threading.Event()
        class DelayedFixture:
            name = "delayed_test_fixture"
            def infer(self, *_):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Test synchronization timed out")
                return proposal()
        service = IntentService(DelayedFixture())
        result = []
        failures = []
        def infer():
            try:
                result.append(service.infer(payload_at(), now=100))
            except Exception as exc:
                failures.append(exc)
        thread = threading.Thread(target=infer, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(BlockingIOError):
                service.infer(payload_at(), now=100)
            self.assertEqual(service.cancel()["status"], "CANCEL_REQUESTED")
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(result[0]["status"], "SUPERSEDED")
        self.assertEqual(service.pending, {})

    def test_cancelled_and_expired_proposals_cannot_commit(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                service = IntentService(ReplayVLM())
                result = service.infer(payload_at(confirmed=False), now=100)
                if cancel:
                    service.cancel()
                now = 100.2 if cancel else 116.0
                with self.assertRaises(ValueError):
                    service.commit(dict(payload_at(now), proposal_id=result["proposal_id"]), now=now)

    def test_scene_and_hand_windows_must_be_aligned(self):
        data = payload_at()
        data["scene"]["captured_at_s"] = 99.7
        with self.assertRaises(ValueError):
            IntentService(ReplayVLM()).infer(data, now=100)

    def test_pending_semantic_cache_is_bounded(self):
        service = IntentService(ReplayVLM())
        for _ in range(30):
            service.infer(payload_at(confirmed=False), now=100)
        self.assertEqual(len(service.pending), 16)


class HTTPGatewayTests(unittest.TestCase):
    def setUp(self):
        self.server = make_server("127.0.0.1", 0, access_token="unit-test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, path, data=None, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer unit-test-token"
        req = urllib.request.Request(self.base + path,
            data=None if data is None else json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.load(response)

    def test_health_and_authenticated_task_flow(self):
        status, health = self.request("/health", authenticated=False)
        self.assertEqual(status, 200)
        self.assertIn("NOT_live", health["provider"])
        now = time.time()
        status, result = self.request("/v1/intent", payload_at(now))
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "TASK_READY")
        status, result = self.request("/v1/cancel", {})
        self.assertEqual((status, result["status"]), (200, "CANCEL_REQUESTED"))

    def test_authentication_and_invalid_input_errors_are_specific(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/intent", payload_at(time.time()), authenticated=False)
        self.assertEqual(caught.exception.code, 401)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/intent", payload_at(100))
        self.assertEqual(caught.exception.code, 422)

    def test_public_bind_without_token_is_rejected_before_listening(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            make_server("0.0.0.0", 0)


class MotionTests(unittest.TestCase):
    def test_mecanum_body_wheel_round_trip(self):
        model = Mecanum()
        for velocity in ((0, 0, 0), (.2, 0, 0), (0, .15, 0), (.1, -.2, .5)):
            with self.subTest(velocity=velocity):
                for actual, expected in zip(model.body(model.wheels(*velocity)), velocity):
                    self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            model.wheels(math.nan, 0, 0)

    def test_grid_path_respects_inflation_and_diagonal_corner_checks(self):
        planner = GridPlanner(((.30, -.10, .50, .20),), resolution=.05, radius=.10)
        route = planner.path((0, 0), (.9, 0))
        self.assertEqual(route[0], (0, 0))
        self.assertEqual(route[-1], (.9, 0))
        self.assertTrue(any(abs(y) > .2 for _, y in route))
        for point in route:
            self.assertFalse(planner.blocked(point))
        for a, b in zip(route, route[1:]):
            if abs(a[0] - b[0]) > 1e-8 and abs(a[1] - b[1]) > 1e-8:
                self.assertFalse(planner.blocked((a[0], b[1])))
                self.assertFalse(planner.blocked((b[0], a[1])))

    def test_blocked_goal_and_disconnected_route_fail(self):
        with self.assertRaises(ValueError):
            GridPlanner(((.4, -.9, .5, 1.1),)).path((0, 0), (1, 0))
        with self.assertRaises(ValueError):
            GridPlanner(((.4, -.1, .6, .1),)).path((0, 0), (.5, 0))

    def test_joint_reference_slew_is_bounded(self):
        self.assertEqual(slew((0, 1, 2), (1, 0, 2.01), .1), (.1, .9, 2.01))


class CameraStreamTests(unittest.TestCase):
    def test_adapter_uses_real_subscribe_read_api_and_remote_utc(self):
        calls = []
        image = SimpleNamespace(shape=(480, 640, 3))
        frame = SimpleNamespace(header={"captured_utc_ns": 100_000_000_000,
                                        "captured_monotonic_ns": 999},
                                sequence=7, image=image)
        class Subscription:
            def read(self, timeout):
                calls.append(("read", timeout))
                return frame
        class Client:
            def __init__(self, endpoint):
                calls.append(("client", endpoint))
            def __enter__(self):
                calls.append(("enter",))
                return self
            def subscribe(self, topic, warm_up_timeout):
                calls.append(("subscribe", topic, warm_up_timeout))
                return Subscription()
            def __exit__(self, *args):
                calls.append(("exit",))
        source = CameraStreamSource("tcp://127.0.0.1:5555", "front/color", "robot", client_factory=Client)
        try:
            packet, returned = source.read(timeout=.25)
            self.assertIs(returned, image)
            self.assertEqual(packet.captured_at_s, 100)
            self.assertEqual(packet.seq, 7)
            self.assertEqual(packet.payload["topic"], "front/color")
            self.assertEqual((packet.payload["width"], packet.payload["height"]), (640, 480))
        finally:
            source.close()
        self.assertIn(("subscribe", "front/color", 5), calls)
        self.assertIn(("read", .25), calls)
        self.assertEqual(calls[-1], ("exit",))

    def test_remote_monotonic_timestamp_cannot_replace_missing_utc(self):
        class Client:
            def __init__(self, *_): pass
            def __enter__(self): return self
            def subscribe(self, *_, **__): return self
            def read(self, **_):
                return SimpleNamespace(header={"captured_monotonic_ns": 1000})
            def __exit__(self, *_): pass
        source = CameraStreamSource("tcp://127.0.0.1:5555", "front/color", "robot", client_factory=Client)
        try:
            with self.assertRaises(ValueError):
                source.read()
        finally:
            source.close()

    def test_no_camera_frame_has_explicit_error(self):
        class Client:
            def __init__(self, *_): pass
            def __enter__(self): return self
            def subscribe(self, *_, **__): return self
            def read(self, **_): return None
            def __exit__(self, *_): pass
        source = CameraStreamSource("tcp://127.0.0.1:5555", "front/color", "robot", client_factory=Client)
        try:
            with self.assertRaises((ValueError, TimeoutError)):
                source.read()
        finally:
            source.close()


if __name__ == "__main__":
    unittest.main()
