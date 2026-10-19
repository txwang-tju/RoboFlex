import copy
from dataclasses import replace
import math
import unittest

from roboflex.backends import MockArm
from roboflex.config import Config
from roboflex.controller import SharedAutonomyController
from roboflex.demo import sample
from roboflex.intent import NO_INTENT, TemporalIntentEstimator
from roboflex.planning import FourAxisIK, GeometricPickPlacePolicy, PlanError, segment_hits_box
from roboflex.types import Glove, Hand, Observation, Target, distance


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config().validate()
        self.policy = GeometricPickPlacePolicy(self.cfg.arm)
        self.arm = MockArm(self.policy.home_joints)
        self.c = SharedAutonomyController(self.cfg, self.arm, self.policy)
        self.t = -0.05

    def advance(self, seconds, transform=lambda o: o, now_offset=0):
        steps = round(seconds / 0.05)
        result = None
        for _ in range(steps):
            self.t = round(self.t + 0.05, 5)
            result = self.c.step(transform(replace(sample(self.t), confirm=False)), self.t + now_offset)
        return result

    def ready(self):
        self.advance(1.5)
        self.assertEqual(self.c.state, "CONFIRMING")

    def start(self):
        self.ready()
        self.advance(0.05, lambda o: replace(o, confirm=True))
        self.assertEqual(self.c.state, "EXECUTING")

    def test_intention_alone_never_starts(self):
        self.advance(4.0)
        self.assertEqual(self.c.state, "CONFIRMING")
        self.assertEqual(self.arm.commands, [])

    def test_success_is_complete_sequence_and_home(self):
        self.start()
        self.advance(13)
        self.assertEqual(self.c.state, "COMPLETE")
        self.assertEqual(self.arm.commands[-1].stage, "return_home")
        self.assertLess(distance(self.arm.joints_deg, self.policy.home_joints), 1e-6)

    def test_ambiguous_never_selects(self):
        self.advance(4, lambda o: replace(sample(o.timestamp, "ambiguous"), confirm=True))
        self.assertEqual(self.c.state, "OBSERVING")
        self.assertEqual(self.arm.commands, [])

    def test_natural_stationary_hand_is_no_intent(self):
        self.advance(3, lambda o: replace(o, hand=replace(o.hand, pointing=False)))
        self.assertEqual(self.c.state, "OBSERVING")
        self.assertIsNone(self.c.intent.target_id)

    def test_early_held_confirm_not_reused(self):
        self.advance(3, lambda o: replace(o, confirm=True))
        self.assertEqual(self.c.state, "CONFIRMING")
        self.assertEqual(self.arm.commands, [])

    def test_new_confirm_after_release_starts(self):
        self.advance(2, lambda o: replace(o, confirm=True))
        self.advance(0.1)
        self.advance(0.05, lambda o: replace(o, confirm=True))
        self.assertEqual(self.c.state, "EXECUTING")

    def test_stale_glove_confirmation_ignored(self):
        self.ready()
        self.advance(1, lambda o: replace(o, glove=Glove(0, (0.3, 0.3), confirm=True)))
        self.assertEqual(self.c.state, "CONFIRMING")
        self.assertFalse(self.c.intent.sensor_used)

    def test_glove_without_visual_hand_cannot_select(self):
        self.advance(2, lambda o: replace(o, hand=None, glove=Glove(o.timestamp, (0.8, 0.8), confirm=True)))
        self.assertEqual(self.c.state, "OBSERVING")

    def test_pinch_changes_ray_but_confirms_locked_target(self):
        self.ready()
        self.advance(0.45, lambda o: replace(o, hand=replace(o.hand, pinch=True, pointing=False)))
        self.assertEqual(self.c.state, "EXECUTING")

    def test_pinch_held_before_selection_must_release(self):
        self.advance(3, lambda o: replace(o, hand=replace(o.hand, pinch=True)))
        self.assertEqual(self.c.state, "CONFIRMING")
        self.assertEqual(self.arm.commands, [])
        self.advance(0.1)
        self.advance(0.5, lambda o: replace(o, hand=replace(o.hand, pinch=True)))
        self.assertEqual(self.c.state, "EXECUTING")

    def test_confirmation_expires(self):
        self.ready()
        for _ in range(120):
            self.advance(0.05)
            if self.c.state != "CONFIRMING":
                break
        self.assertEqual(self.c.state, "OBSERVING")
        self.assertIn("confirmation_expired", self.c.message)

    def test_cancel_latched_no_more_commands(self):
        self.start()
        self.advance(1)
        self.advance(0.05, lambda o: replace(o, cancel=True))
        count = len(self.arm.commands)
        self.advance(4, lambda o: replace(o, confirm=True))
        self.assertEqual(self.c.state, "CANCELLED")
        self.assertEqual(len(self.arm.commands), count)

    def test_cancel_after_gripping_does_not_drop_object(self):
        self.start()
        self.advance(4)
        self.assertEqual(self.arm.gripper_mm, 35)
        self.advance(0.05, lambda o: replace(o, cancel=True))
        self.assertEqual(self.arm.gripper_mm, 35)
        self.assertEqual(self.c.state, "CANCELLED")

    def test_retraction_before_confirmation_cancels(self):
        self.advance(2)
        for i in range(1, 8):
            dx = i * 0.03
            self.advance(0.05, lambda o, dx=dx: replace(o, hand=Hand((0.2-dx, 0.4), (0.35-dx, 0.4), (0.25-dx, 0.4))))
        self.assertEqual(self.c.state, "CANCELLED")

    def test_stale_camera_frame_stops(self):
        self.start()
        self.advance(0.05, now_offset=1.0)
        self.assertEqual(self.c.state, "CANCELLED")
        self.assertEqual(self.c.message, "stale_observation")

    def test_observation_gap_requires_reconfirmation(self):
        self.ready()
        self.t += 0.8
        self.c.step(replace(sample(self.t), confirm=True))
        self.assertEqual(self.c.state, "OBSERVING")
        self.assertEqual(self.arm.commands, [])

    def test_calibrated_pose_disappears_before_grasp(self):
        self.start()
        self.advance(0.05, lambda o: replace(o, targets=(replace(o.targets[0], position_mm=None), o.targets[1])))
        self.assertEqual(self.c.state, "CANCELLED")
        self.assertIn("pose_lost", self.c.message)

    def test_out_of_order_frame_faults(self):
        self.start()
        self.c.step(sample(self.t - 0.1), self.t)
        self.assertEqual(self.c.state, "FAULT")

    def test_hand_loss_stops(self):
        self.start()
        self.advance(2, lambda o: replace(o, hand=None))
        self.assertEqual(self.c.state, "CANCELLED")

    def test_target_change_does_not_silently_retarget(self):
        self.start()
        self.advance(0.05, lambda o: replace(o, targets=(replace(o.targets[0], position_mm=(280, -70, 35)), o.targets[1])))
        self.assertEqual(self.c.state, "CANCELLED")
        self.assertEqual(self.c.message, "target_moved_reconfirm_required")

    def test_bad_plan_sends_no_move(self):
        self.ready()
        bad = Target("0", "cup", (0.65, 0.40), None)
        self.advance(0.05, lambda o: replace(o, targets=(bad, o.targets[1]), confirm=True))
        self.assertEqual(self.c.state, "FAULT")
        self.assertEqual(self.arm.commands, [])

    def test_estop_and_explicit_reset_no_automatic_restart(self):
        self.start()
        self.advance(0.05, lambda o: replace(o, estop=True))
        self.advance(1, lambda o: replace(o, confirm=True))
        self.assertEqual(self.c.state, "ESTOP")
        self.advance(0.05, lambda o: replace(o, reset=True))
        self.assertEqual(self.c.state, "OBSERVING")
        self.assertEqual(self.arm.status, "STOPPED")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config().validate()
        self.p = GeometricPickPlacePolicy(self.cfg.arm)

    def test_inverse_forward_round_trip(self):
        for point in ((170, 0, 180), (210, -70, 35), (180, 100, 35), (220, 30, 150)):
            self.assertLess(distance(self.p.ik.forward(self.p.ik.inverse(point)), point), 1e-7)

    def test_unreachable_ik_rejected(self):
        with self.assertRaises(PlanError):
            self.p.ik.inverse((1000, 0, 100))

    def test_complete_plan_limits_and_joint_speed(self):
        plan = self.p.plan(sample(0).targets[0], self.p.home_joints)
        self.assertGreater(len(plan.commands), 10)
        previous = self.p.home_joints
        for command in plan.commands:
            if command.joints_deg is not None:
                self.assertTrue(all(lo <= v <= hi for v, (lo, hi) in zip(command.joints_deg, self.cfg.arm.joint_limits_deg)))
                self.assertLessEqual(max(abs(a - b) for a, b in zip(previous, command.joints_deg)) / command.duration_s, self.cfg.arm.max_joint_speed_deg_s + 1e-6)
                previous = command.joints_deg

    def test_wrong_start_pose_rejected(self):
        with self.assertRaises(PlanError):
            self.p.plan(sample(0).targets[0], (0, 0, 0, 0))

    def test_nonfinite_pose_rejected(self):
        with self.assertRaises(PlanError):
            self.p.ik.inverse((float("nan"), 0, 0))
        with self.assertRaises(ValueError):
            Target("bad", "bad", (0.5, 0.5), (math.inf, 0, 0))

    def test_thin_keepout_between_samples_caught(self):
        self.assertTrue(segment_hits_box((0, 0, 0), (100, 0, 0), [[48.1, -1, -1], [48.2, 1, 1]]))
        self.cfg.arm.keepouts_mm = [[[185, -100, 140], [195, 10, 160]]]
        with self.assertRaises(PlanError):
            self.p.plan(sample(0).targets[0], self.p.home_joints)

    def test_invalid_configuration_rejected(self):
        self.cfg.arm.link1_mm = -10
        with self.assertRaises(ValueError):
            self.cfg.validate()


class IntentTests(unittest.TestCase):
    def test_duplicate_timestamp_rejected(self):
        estimator = TemporalIntentEstimator(Config().intent)
        estimator.update(sample(0))
        with self.assertRaises(ValueError):
            estimator.update(sample(0))

    def test_disappeared_target_probability_transfers_to_none(self):
        estimator = TemporalIntentEstimator(Config().intent)
        for i in range(50):
            estimator.update(sample(i * 0.05))
        result = estimator.update(Observation(2.5))
        self.assertEqual(result.probabilities, {NO_INTENT: 1.0})

    def test_time_based_smoothing_independent_of_fps(self):
        results = []
        for hz in (20, 50):
            estimator = TemporalIntentEstimator(Config().intent)
            for i in range(hz * 2):
                result = estimator.update(sample(i / hz))
            results.append(result.score)
        self.assertAlmostEqual(*results, places=3)


if __name__ == "__main__":
    unittest.main()
