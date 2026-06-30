import unittest

import numpy as np

from src.pitch_keypoints import SoccerPitchKeypointConfig
from src.vision_pipeline import FootballVisionPipeline


class SoccerPitchKeypointConfigTests(unittest.TestCase):
    def test_roboflow_template_has_expected_shape_and_order(self):
        config = SoccerPitchKeypointConfig()

        self.assertEqual(len(config.vertices), 32)
        self.assertEqual(len(config.labels), 32)
        self.assertEqual(len(config.edges), 33)
        self.assertEqual(config.labels[:6], ["01", "02", "03", "04", "05", "06"])
        self.assertEqual(config.labels[-2:], ["14", "19"])

        vertices = config.vertices_array()
        np.testing.assert_allclose(vertices[0], [0.0, 0.0])
        np.testing.assert_allclose(vertices[13], [60.0, 0.0])
        np.testing.assert_allclose(vertices[30], [50.85, 35.0])
        np.testing.assert_allclose(vertices[31], [69.15, 35.0])

    def test_vertices_convert_to_project_world_coordinates(self):
        config = SoccerPitchKeypointConfig()
        world = config.world_vertices_array()

        np.testing.assert_allclose(world[0], [-52.5, 34.0])
        np.testing.assert_allclose(world[29], [52.5, -34.0])
        np.testing.assert_allclose(world[30], [-8.00625, 0.0], atol=1e-4)
        np.testing.assert_allclose(world[31], [8.00625, 0.0], atol=1e-4)

    def test_visible_indices_respect_confidence_threshold(self):
        config = SoccerPitchKeypointConfig()
        keypoints = [(0.0, 0.0, 0.1), (1.0, 1.0, 0.8), (2.0, 2.0, 0.5)]

        self.assertEqual(config.visible_indices(keypoints, 0.5), [1, 2])

    def test_homography_estimation_reports_valid_and_inlier_indices(self):
        pipeline = FootballVisionPipeline(object_model_path=None, keypoint_confidence=0.5)
        config = SoccerPitchKeypointConfig()
        world = config.world_vertices_array()
        image_points = np.column_stack([world[:, 0] * 10.0 + 960.0, world[:, 1] * -10.0 + 540.0])
        keypoints = [(float(x), float(y), 0.9) for x, y in image_points]

        H, valid_count, inliers, error, inlier_indices = pipeline._estimate_homography_from_keypoints(keypoints)

        self.assertIsNotNone(H)
        self.assertEqual(valid_count, 32)
        self.assertEqual(inliers, 32)
        self.assertEqual(inlier_indices, list(range(32)))
        self.assertLess(error, 1e-4)
if __name__ == "__main__":
    unittest.main()
