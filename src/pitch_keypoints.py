"""Roboflow-style soccer pitch keypoint configuration.

The keypoint order mirrors the common 32-point football pitch pose model used by
Roboflow Sports examples, while coordinates are scaled to this project's
105m x 68m world model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


Point2D = Tuple[float, float]
Edge = Tuple[int, int]
Keypoint = Tuple[float, float, float]


@dataclass(frozen=True)
class SoccerPitchKeypointConfig:
    """32-point pitch template plus debug graph metadata."""

    length: float = 105.0
    width: float = 68.0
    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08",
        "09", "10", "11", "12", "13", "15", "16", "17",
        "18", "20", "21", "22", "23", "24", "25", "26",
        "27", "28", "29", "30", "31", "32", "14", "19",
    ])
    vertices: List[Point2D] = field(default_factory=lambda: [
        (0.0, 0.0), (0.0, 13.84), (0.0, 24.84), (0.0, 43.16), (0.0, 54.16), (0.0, 68.0),
        (5.5, 24.84), (5.5, 43.16), (11.0, 34.0),
        (16.5, 13.84), (16.5, 24.84), (16.5, 43.16), (16.5, 54.16),
        (52.5, 0.0), (52.5, 24.84), (52.5, 43.16), (52.5, 68.0),
        (88.5, 13.84), (88.5, 24.84), (88.5, 43.16), (88.5, 54.16),
        (94.0, 34.0),
        (99.5, 24.84), (99.5, 43.16),
        (105.0, 0.0), (105.0, 13.84), (105.0, 24.84), (105.0, 43.16), (105.0, 54.16), (105.0, 68.0),
        (43.35, 34.0), (61.65, 34.0),
    ])
    edges: List[Edge] = field(default_factory=lambda: [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
        (0, 13), (1, 9), (2, 6), (3, 7), (4, 12), (5, 16),
        (13, 24), (17, 25), (22, 26), (23, 27), (20, 28), (16, 29),
    ])

    def vertices_array(self) -> np.ndarray:
        return np.array(self.vertices, dtype=np.float32)

    def world_vertices_array(self) -> np.ndarray:
        vertices = self.vertices_array()
        return np.column_stack([
            vertices[:, 0] - self.length / 2.0,
            self.width / 2.0 - vertices[:, 1],
        ]).astype(np.float32)

    def visible_indices(self, keypoints: Sequence[Keypoint], threshold: float) -> List[int]:
        return [idx for idx, (_, _, confidence) in enumerate(keypoints) if confidence >= threshold]
