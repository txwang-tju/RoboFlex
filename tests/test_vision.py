import importlib.util
import os
import unittest

AVAILABLE = bool(importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"))


@unittest.skipUnless(AVAILABLE, "Install vision extras for optical geometry tests")
class OpticalGeometryTests(unittest.TestCase):
    def calibration(self):
        return {"camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
                "T_base_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "image_size": [640, 480]}

    def test_synthetic_marker_pose_with_known_projection(self):
        import cv2
        import numpy as np
        from roboflex.vision import MarkerPose
        pose = MarkerPose(self.calibration())
        points = np.array([[-20, 20, 0], [20, 20, 0], [20, -20, 0], [-20, -20, 0]], dtype=float)
        rvec = np.array([2.9, 0.1, 0.05])
        tvec = np.array([30., 20., 500.])
        pixels, _ = cv2.projectPoints(points, rvec, tvec, pose.K, pose.D)
        result = pose.position(pixels, {"marker_size_mm": 40, "marker_to_grasp_mm": [0, 0, 0]}, (640, 480))
        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result, tvec, atol=1e-4), result)

    def test_wrong_resolution_rejected(self):
        from roboflex.vision import MarkerPose
        with self.assertRaises(ValueError):
            MarkerPose(self.calibration()).position([], {}, (1280, 720))

    def test_invalid_rigid_transform_rejected(self):
        from roboflex.vision import MarkerPose
        c = self.calibration()
        c["T_base_camera"][0][0] = 2
        with self.assertRaises(ValueError):
            MarkerPose(c)

    @unittest.skipUnless(os.environ.get("ROBOFLEX_TEST_MODEL"), "Set ROBOFLEX_TEST_MODEL for actual MediaPipe inference smoke test")
    def test_real_hand_model_and_aruco_on_blank_fixture_image(self):
        import cv2
        import numpy as np
        from roboflex.config import Config
        from roboflex.vision import WebcamPerception
        model = WebcamPerception(Config(), os.environ["ROBOFLEX_TEST_MODEL"])
        try:
            image = np.full((480, 640, 3), 255, dtype=np.uint8)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            marker = cv2.aruco.generateImageMarker(dictionary, 0, 100)
            image[100:200, 400:500] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            obs = model.observe(image, 1.0)
            self.assertIsNone(obs.hand)
            self.assertEqual(len(obs.targets), 1)
            self.assertEqual(obs.targets[0].id, "0")
            self.assertEqual(tuple(obs.targets[0].position_mm), (210, -70, 35))
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
