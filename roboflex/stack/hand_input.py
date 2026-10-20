"""Offline real-image hand descriptor. Images feed MediaPipe, sensors stay separate."""
from __future__ import annotations
import json
from pathlib import Path
from .perception import hand_features, structured_hand_text


def describe_image(image_path: str, model_path: str, glove_path: str) -> dict:
    import mediapipe as mp
    image = mp.Image.create_from_file(str(Path(image_path).resolve()))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path), num_hands=1)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(image)
    visual = hand_features([[p.x, p.y, p.z] for p in result.hand_landmarks[0]]) if result.hand_landmarks else {
        "source": "mediapipe_image_landmarks", "hand_present": False}
    glove = json.loads(Path(glove_path).read_text(encoding="utf-8"))
    return {"mode": "offline_descriptor_NOT_dispatch", "image": Path(image_path).name,
            "hand_text": structured_hand_text(visual, glove),
            "note": "Sensor calibration/units must be supplied; no world geometry inferred"}
