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
    """32-point pitch template plus debug graph metadata.

    The model was trained with Roboflow Sports' 120m x 70m canonical template.
    ``world_vertices_array`` scales that template into this project's 105m x 68m
    render coordinate system. Keeping the trained template proportions matters:
    replacing it directly with 105m FIFA dimensions shifts penalty-area players.
    """

    length: float = 105.0
    width: float = 68.0
    template_length: float = 120.0
    template_width: float = 70.0
    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08",
        "09", "10", "11", "12", "13", "15", "16", "17",
        "18", "20", "21", "22", "23", "24", "25", "26",
        "27", "28", "29", "30", "31", "32", "14", "19",
    ])
    vertices: List[Point2D] = field(default_factory=lambda: [
        (0.0, 0.0), (0.0, 14.5), (0.0, 25.84), (0.0, 44.16), (0.0, 55.5), (0.0, 70.0),
        (5.5, 25.84), (5.5, 44.16), (11.0, 35.0),
        (20.15, 14.5), (20.15, 25.84), (20.15, 44.16), (20.15, 55.5),
        (60.0, 0.0), (60.0, 25.85), (60.0, 44.15), (60.0, 70.0),
        (99.85, 14.5), (99.85, 25.84), (99.85, 44.16), (99.85, 55.5),
        (109.0, 35.0),
        (114.5, 25.84), (114.5, 44.16),
        (120.0, 0.0), (120.0, 14.5), (120.0, 25.84), (120.0, 44.16), (120.0, 55.5), (120.0, 70.0),
        (50.85, 35.0), (69.15, 35.0),
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
        scaled_x = vertices[:, 0] * (self.length / self.template_length)
        scaled_y = vertices[:, 1] * (self.width / self.template_width)
        return np.column_stack([
            scaled_x - self.length / 2.0,
            self.width / 2.0 - scaled_y,
        ]).astype(np.float32)

    def visible_indices(self, keypoints: Sequence[Keypoint], threshold: float) -> List[int]:
        return [idx for idx, (_, _, confidence) in enumerate(keypoints) if confidence >= threshold]
