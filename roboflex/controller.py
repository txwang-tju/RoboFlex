"""Shared-autonomy controller. Selecting a candidate never alone starts motion."""
from __future__ import annotations

from .backends import ArmBackend
from .config import Config
from .intent import IntentEstimator, NO_INTENT, TemporalIntentEstimator
from .planning import ExecutionPolicy, PlanError
from .types import Decision, Intent, Observation, Target, distance


class SharedAutonomyController:
    TERMINAL = {"COMPLETE", "CANCELLED", "FAULT", "ESTOP"}

    def __init__(self, config: Config, backend: ArmBackend, policy: ExecutionPolicy,
                 estimator: IntentEstimator | None = None):
        self.cfg, self.arm, self.policy = config, backend, policy
        self.estimator = estimator or TemporalIntentEstimator(config.intent)
        self.state = "OBSERVING"
        self.message = "waiting_for_intent"
        self.intent = Intent({NO_INTENT: 1.0}, None, 1.0, 1.0, "no_intent")
        self.candidate: str | None = None
        self.candidate_since: float | None = None
        self.confirm_since = 0.0
        self.snapshot: Target | None = None
        self.execution_deadline = 0.0
        self.last_hand_time: float | None = None
        self.last_timestamp: float | None = None
        self.last_confirm = False
        self.last_reset = False
        self.pinch_armed = False
        self.pinch_since: float | None = None

    def _stop(self, state: str, reason: str):
        self.arm.stop(reason)
        self.state, self.message = state, reason
        self.candidate_since = None
        self.pinch_since = None

    def _observe(self, reason: str):
        self.state, self.message = "OBSERVING", reason
        self.candidate = None
        self.candidate_since = None
        self.snapshot = None
        self.pinch_since = None

    def step(self, obs: Observation, now: float | None = None) -> Decision:
        now = obs.timestamp if now is None else now
        # Events are edge-triggered. A held/stale confirmation cannot start a later task.
        sensor = obs.glove
        fresh_sensor = bool(sensor and sensor.enabled and 0 <= now - sensor.timestamp <= self.cfg.intent.sensor_max_age_s)
        confirm = obs.confirm or bool(fresh_sensor and sensor and sensor.confirm)
        confirm_edge = confirm and not self.last_confirm
        reset_edge = obs.reset and not self.last_reset
        self.last_confirm, self.last_reset = confirm, obs.reset

        def result():
            return Decision(now, self.state, self.snapshot.id if self.snapshot else self.candidate, self.message, self.intent)

        if obs.estop:
            self._stop("ESTOP", "operator_estop")
            return result()
        if self.last_timestamp is not None and obs.timestamp <= self.last_timestamp:
            if self.state == "EXECUTING":
                self._stop("FAULT", "out_of_order_observation")
            else:
                self._observe("out_of_order_observation") if self.state not in self.TERMINAL else None
            return result()
        gap = self.last_timestamp is not None and obs.timestamp - self.last_timestamp > self.cfg.control.max_observation_age_s
        self.last_timestamp = obs.timestamp
        if gap and self.state in {"CONFIRMING", "EXECUTING"}:
            if self.state == "EXECUTING":
                self._stop("CANCELLED", "observation_gap")
            else:
                self.estimator.reset()
                self._observe("observation_gap_reconfirm_required")
            return result()
        if not 0 <= now - obs.timestamp <= self.cfg.control.max_observation_age_s:
            if self.state == "EXECUTING":
                self._stop("CANCELLED", "stale_observation")
            elif self.state not in self.TERMINAL:
                self.estimator.reset()
                self._observe("stale_observation")
            return result()
        if self.state in self.TERMINAL:
            if reset_edge and self.arm.poll(now) in {"IDLE", "STOPPED", "COMPLETE"}:
                self.estimator.reset()
                self._observe("operator_reset")
            return result()
        self.intent = self.estimator.update(obs)
        if obs.hand is not None and obs.hand.quality >= 0.5:
            self.last_hand_time = now
        if obs.cancel or (fresh_sensor and sensor and sensor.cancel) or self.intent.withdrawn:
            if self.state in {"CONFIRMING", "EXECUTING"}:
                self._stop("CANCELLED", "user_cancel_or_withdraw")
            else:
                self.estimator.reset()
                self._observe("user_cancel")
            return result()
        targets = {t.id: t for t in obs.targets}
        if self.state == "EXECUTING":
            if self.last_hand_time is None or now - self.last_hand_time > self.cfg.control.hand_loss_timeout_s:
                self._stop("CANCELLED", "hand_tracking_lost")
                return result()
            assert self.snapshot is not None
            seen = targets.get(self.snapshot.id)
            # Once the arm is near/gripping the object its own motion/occlusion is expected.
            # Backend stage gates this check; it is not a human collision detector.
            stage = getattr(self.arm, "stage", None)
            if stage is None and getattr(self.arm, "plan", None) is not None:
                p = self.arm.plan
                stage = p.commands[min(self.arm.index, len(p.commands) - 1)].stage
            if stage in {"open_before_pick", "home_clearance", "above_target"}:
                if seen is None:
                    self._stop("CANCELLED", "target_lost_before_grasp")
                    return result()
                if self.snapshot.position_mm is not None and seen.position_mm is None:
                    self._stop("CANCELLED", "target_pose_lost_before_grasp")
                    return result()
                shifted = distance(seen.uv, self.snapshot.uv) > self.cfg.control.max_uv_shift
                if seen.position_mm is not None and self.snapshot.position_mm is not None:
                    shifted |= distance(seen.position_mm, self.snapshot.position_mm) > self.cfg.control.max_target_shift_mm
                if shifted:
                    self._stop("CANCELLED", "target_moved_reconfirm_required")
                    return result()
            self.arm.heartbeat(now)
            status = self.arm.poll(now)
            if status == "COMPLETE":
                self.state, self.message = "COMPLETE", "pick_place_sequence_completed"
            elif status in {"FAULT", "STOPPED"}:
                self._stop("FAULT", self.arm.error or "backend_stopped")
            elif now > self.execution_deadline:
                self._stop("FAULT", "execution_timeout")
            return result()

        strong = (self.intent.target_id in targets and self.intent.score >= self.cfg.intent.select_score
                  and self.intent.margin >= self.cfg.intent.select_margin and obs.hand is not None and obs.hand.quality >= 0.5)
        if self.state == "CONFIRMING":
            assert self.snapshot is not None
            pinching = bool(obs.hand and obs.hand.pinch)
            # Pinch changes the pointing geometry. Preserve the already-selected ID
            # briefly while confirming; never switch it to a competing strong target.
            competing = (self.intent.target_id is not None and self.intent.target_id != self.snapshot.id
                         and self.intent.score >= self.cfg.intent.select_score)
            pinch_confirmation_window = pinching and obs.hand.quality >= 0.5 and self.snapshot.id in targets and not competing
            if not ((strong and self.intent.target_id == self.snapshot.id) or pinch_confirmation_window):
                self._observe("candidate_changed_or_uncertain")
                return result()
            if now - self.confirm_since > self.cfg.control.confirmation_timeout_s:
                self._observe("confirmation_expired")
                self.estimator.reset()
                return result()
            seen = targets[self.snapshot.id]
            if distance(seen.uv, self.snapshot.uv) > self.cfg.control.max_uv_shift:
                self._observe("candidate_moved_reconfirm_required")
                return result()
            pinch = bool(obs.hand and obs.hand.pinch)
            if not pinch:
                self.pinch_armed, self.pinch_since = True, None
            elif self.pinch_armed:
                self.pinch_since = self.pinch_since if self.pinch_since is not None else now
            pinch_confirmed = self.pinch_since is not None and now - self.pinch_since >= self.cfg.intent.pinch_hold_s
            if confirm_edge or pinch_confirmed:
                try:
                    # Use fresh position only after preserving identity; target jumps require new confirmation.
                    if seen.position_mm is not None and self.snapshot.position_mm is not None and distance(seen.position_mm, self.snapshot.position_mm) > self.cfg.control.max_target_shift_mm:
                        self._observe("candidate_pose_moved_reconfirm_required")
                        return result()
                    plan = self.policy.plan(seen, self.arm.joints_deg)
                    self.arm.start(plan, now)
                    self.snapshot = seen
                    self.execution_deadline = now + plan.duration_s + self.cfg.control.command_timeout_s * (len(plan.commands) + 1)
                    self.state, self.message = "EXECUTING", "confirmed_pick_place"
                except (PlanError, RuntimeError, OSError) as exc:
                    self._stop("FAULT", f"plan_or_driver_rejected: {exc}")
            return result()
        if not strong:
            self.candidate, self.candidate_since = None, None
            self.message = "ambiguous_or_no_intent"
        else:
            if self.candidate != self.intent.target_id:
                self.candidate = self.intent.target_id
                self.candidate_since = now
            if self.candidate_since is not None and now - self.candidate_since >= self.cfg.intent.dwell_s:
                self.snapshot = targets[self.candidate]
                self.state, self.message, self.confirm_since = "CONFIRMING", "confirm_selected_target", now
                self.pinch_armed = bool(obs.hand and not obs.hand.pinch)
                self.pinch_since = None
            else:
                self.message = "accumulating_temporal_evidence"
        return result()
