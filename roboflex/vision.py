"""Optional real webcam input: MediaPipe Tasks + ArUco fixture objects.

ArUco deliberately provides inspectable object IDs before introducing learned
detectors. Hand Landmarker is a real model; intent inference remains our baseline.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time

from .config import Config
from .types import Hand, Observation, Target, distance


def _deps():
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Camera mode needs: python -m pip install -e '.[vision]'") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco missing; install opencv-contrib-python, avoid multiple OpenCV wheels")
    return cv2, np, mp


class MarkerPose:
    def __init__(self, calibration: dict):
        import numpy as np
        self.K = np.asarray(calibration["camera_matrix"], dtype=float)
        self.D = np.asarray(calibration["dist_coeffs"], dtype=float)
        self.T = np.asarray(calibration["T_base_camera"], dtype=float)
        if self.K.shape != (3, 3) or self.T.shape != (4, 4) or not all(np.isfinite(v).all() for v in (self.K, self.D, self.T)):
            raise ValueError("Invalid camera calibration matrices")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError("Invalid intrinsic camera matrix")
        R = self.T[:3, :3]
        if not np.allclose(self.T[3], [0, 0, 0, 1]) or not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1, atol=1e-3):
            raise ValueError("T_base_camera must be a rigid, right-handed transform")
        self.image_size = tuple(calibration["image_size"])
        self.max_error_px = float(calibration.get("max_reprojection_error_px", 2.0))
        if len(self.image_size) != 2 or min(self.image_size) <= 0 or not math.isfinite(self.max_error_px) or self.max_error_px <= 0:
            raise ValueError("Invalid calibration image size/error threshold")

    def position(self, corners, item: dict, image_size: tuple[int, int]):
        import cv2
        import numpy as np
        if tuple(image_size) != self.image_size:
            raise ValueError("Camera resolution differs from calibrated image_size")
        size = float(item["marker_size_mm"])
        offset = np.asarray(item["marker_to_grasp_mm"], dtype=float)
        if size <= 0 or not math.isfinite(size) or offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Invalid marker size/grasp offset")
        half = size / 2
        points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
        pixels = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(points, pixels, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
            return None
        reprojection, _ = cv2.projectPoints(points, rvec, tvec, self.K, self.D)
        error = float(np.sqrt(np.mean((reprojection.reshape(4, 2) - pixels) ** 2)))
        if not math.isfinite(error) or error > self.max_error_px:
            return None
        R, _ = cv2.Rodrigues(rvec)
        point_camera = R @ offset + tvec.ravel()
        point_base = self.T @ np.append(point_camera, 1.0)
        if not np.isfinite(point_base).all():
            return None
        return tuple(float(v) for v in point_base[:3])


class WebcamPerception:
    def __init__(self, config: Config, model_path: str, hand_label: str = "Right", *, simulated: bool = True):
        cv2, np, mp = _deps()
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}; run download-model first")
        self.cv2, self.np, self.mp, self.cfg = cv2, np, mp, config
        self.hand_label, self.simulated = hand_label, simulated
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6)
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(options)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.pose = MarkerPose(config.camera_calibration) if config.camera_calibration else None
        if not simulated and self.pose is None:
            self.model.close()
            raise ValueError("Physical camera control requires camera intrinsic/extrinsic calibration")
        self.last_model_ms = -1

    def observe(self, bgr, timestamp: float, glove=None) -> Observation:
        cv2, np, mp = self.cv2, self.np, self.mp
        height, width = bgr.shape[:2]
        corners, ids, _ = self.detector.detectMarkers(bgr)
        # Do NOT mirror coordinates; any mirrored display must be separate from inference.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self.last_model_ms + 1, int(timestamp * 1000))
        self.last_model_ms = timestamp_ms
        result = self.model.detect_for_video(image, timestamp_ms)
        matching = [lm for lm, handedness in zip(result.hand_landmarks, result.handedness)
                    if handedness and handedness[0].category_name == self.hand_label]
        hand = None
        if len(matching) == 1:
            lm = matching[0]
            p = lambda i: (lm[i].x, lm[i].y)
            palm_width = max(distance(p(5), p(17)), 1e-5)
            pinch = distance(p(4), p(8)) / palm_width < 0.30
            pointing = distance(p(8), p(0)) > distance(p(6), p(0)) * 1.12
            # Presence was gated by task thresholds; handedness score is NOT tracking quality.
            hand = Hand(p(0), p(8), p(5), pointing, pinch, 1.0,
                        tuple((point.x, point.y, point.z) for point in lm))
            for point in (p(0), p(5), p(8), p(4)):
                cv2.circle(bgr, (int(point[0] * width), int(point[1] * height)), 4, (0, 220, 120), -1)
            cv2.line(bgr, (int(p(5)[0] * width), int(p(5)[1] * height)),
                     (int(p(8)[0] * width), int(p(8)[1] * height)), (0, 180, 255), 2)
        targets = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
            for square, marker_id in zip(corners, ids.ravel()):
                key = str(int(marker_id))
                item = self.cfg.targets.get(key)
                if item is None:
                    continue
                centre = np.asarray(square).reshape(4, 2).mean(axis=0)
                position = None
                if self.pose:
                    position = self.pose.position(square, item, (width, height))
                elif self.simulated:
                    # Explicit mock fixture coordinates; never used for physical execution.
                    position = item.get("position_mm")
                targets.append(Target(key, item.get("label", key), (float(centre[0] / width), float(centre[1] / height)),
                                      position, 1.0, float(item.get("grip_mm", 25))))
        return Observation(timestamp, tuple(targets), hand, glove, image_aspect=width / height)

    def close(self):
        self.model.close()


class LatestCamera:
    """One-slot capture queue; a stalled camera cannot repeat an old frame as fresh."""
    def __init__(self, index: int, width: int, height: int):
        import cv2
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open camera {index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest = None
        self.lock = threading.Lock()
        self.quit = threading.Event()
        self.thread = threading.Thread(target=self._run, name="roboflex-camera", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.quit.is_set():
            ok, frame = self.capture.read()
            with self.lock:
                self.latest = (time.monotonic(), frame) if ok else None
            if not ok:
                self.quit.wait(0.05)

    def get(self):
        with self.lock:
            value, self.latest = self.latest, None
            return value

    def close(self):
        self.quit.set()
        self.thread.join(timeout=0.2)
        self.capture.release()
