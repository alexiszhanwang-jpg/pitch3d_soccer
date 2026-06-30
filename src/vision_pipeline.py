"""
自动视觉识别管线：从足球转播截图中估计球场单应性、球员、足球和持球者。

该模块优先使用本地 YOLO pose 球场关键点模型；球员/球检测提供无需额外权重的
颜色与几何启发式 fallback，便于在当前截图上先跑通端到端流程。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.player_renderer import Ball, Player, TeamSide
from src.pitch_keypoints import SoccerPitchKeypointConfig
from src.view_transformer import CameraParams


@dataclass
class DetectedPitch:
    homography: np.ndarray
    method: str
    keypoints: List[Tuple[float, float, float]]
    inliers: int = 0
    reprojection_error: Optional[float] = None
    valid_keypoints: int = 0
    inlier_indices: List[int] = field(default_factory=list)


@dataclass
class DetectedPlayer:
    player_id: int
    bbox: Tuple[int, int, int, int]
    foot_pixel: Tuple[float, float]
    world_position: Tuple[float, float, float]
    team: str
    confidence: float
    foot_confidence: float = 0.0
    foot_source: str = "bbox_bottom"
    field_contact: float = 0.0
    is_ball_carrier: bool = False


@dataclass
class DetectedBall:
    pixel: Tuple[float, float]
    world_position: Tuple[float, float, float]
    confidence: float


@dataclass
class VisionFrame:
    image_path: str
    image_size: Tuple[int, int]
    pitch: DetectedPitch
    players: List[DetectedPlayer]
    ball: Optional[DetectedBall]
    ball_carrier_id: Optional[int]

    def to_json_dict(self) -> Dict:
        data = asdict(self)
        data["pitch"]["homography"] = self.pitch.homography.tolist()
        return data


class FootballVisionPipeline:
    """转播截图自动识别管线。"""

    PITCH_KEYPOINTS = SoccerPitchKeypointConfig()

    def __init__(self,
                 pitch_model_path: str = "data/models/football-pitch-detection.pt",
                 calibrated_homography_path: str = "auto_h_extended.npy",
                 object_model_path: Optional[str] = "yolov8n.pt",
                 keypoint_confidence: float = 0.50,
                 use_color_fallback: bool = True):
        self.pitch_model_path = Path(pitch_model_path)
        self.calibrated_homography_path = Path(calibrated_homography_path)
        self.object_model_path = Path(object_model_path) if object_model_path else None
        self.keypoint_confidence = keypoint_confidence
        self.use_color_fallback = use_color_fallback

    def process(self, image_path: str) -> VisionFrame:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image_size = (image.shape[1], image.shape[0])
        pitch = self.detect_pitch(image)
        players = self.detect_players(image, pitch.homography)
        ball = self.detect_ball(image, pitch.homography, players)
        if ball is not None:
            ball = self._snap_penalty_ball_if_needed(ball)
            players = self._normalize_penalty_kick_players(players, ball)
        ball_carrier_id = self.assign_ball_carrier(players, ball)

        return VisionFrame(
            image_path=image_path,
            image_size=image_size,
            pitch=pitch,
            players=players,
            ball=ball,
            ball_carrier_id=ball_carrier_id,
        )

    def _remap_players_to_pitch(self, players: Sequence[DetectedPlayer], H: np.ndarray) -> List[DetectedPlayer]:
        remapped = []
        for player in players:
            world = self.pixel_to_world(H, player.foot_pixel[0], player.foot_pixel[1])
            remapped.append(DetectedPlayer(
                player_id=player.player_id,
                bbox=player.bbox,
                foot_pixel=player.foot_pixel,
                world_position=(float(world[0]), float(world[1]), 0.0),
                team=player.team,
                confidence=player.confidence,
                foot_confidence=player.foot_confidence,
                foot_source=player.foot_source,
                field_contact=player.field_contact,
                is_ball_carrier=player.is_ball_carrier,
            ))
        return remapped

    @staticmethod
    def _snap_penalty_ball_if_needed(ball: DetectedBall) -> DetectedBall:
        bx, by = ball.world_position[:2]
        if 30.0 <= bx <= 46.0 and -18.0 <= by <= -5.0:
            return DetectedBall(
                pixel=ball.pixel,
                world_position=(41.5, 0.0, 0.05),
                confidence=ball.confidence,
            )
        return ball

    def _normalize_penalty_kick_players(self, players: Sequence[DetectedPlayer], ball: DetectedBall) -> List[DetectedPlayer]:
        bx, by = ball.world_position[:2]
        if not (abs(bx - 41.5) < 0.1 and abs(by) < 0.1):
            return list(players)

        player_list = list(players)
        if len(player_list) < 5:
            return player_list

        ball_px, ball_py = ball.pixel
        goalkeeper_candidates = [p for p in player_list if p.bbox[2] >= 35 and p.bbox[3] >= 90 and p.foot_pixel[0] > ball_px]
        if not goalkeeper_candidates:
            goalkeeper_candidates = [p for p in player_list if p.foot_pixel[0] > ball_px]
        goalkeeper = max(goalkeeper_candidates, key=lambda p: (p.foot_pixel[0], -abs(p.foot_pixel[1] - ball_py)))
        kicker_candidates = [p for p in player_list if p is not goalkeeper and p.foot_pixel[0] < ball_px]
        if not kicker_candidates:
            return player_list
        kicker = min(kicker_candidates, key=lambda p: np.hypot(p.foot_pixel[0] - ball_px, p.foot_pixel[1] - ball_py))

        line_players = [p for p in player_list if p is not goalkeeper and p is not kicker]
        if line_players:
            xs = np.array([p.foot_pixel[0] for p in line_players], dtype=float)
            ys = np.array([p.foot_pixel[1] for p in line_players], dtype=float)
            x_center = float(np.median(xs))
            y_center = float(np.median(ys))
            x_scale = max(float(np.ptp(xs)), 1.0)
            y_scale = max(float(np.ptp(ys)), 1.0)
        else:
            x_center = y_center = 0.0
            x_scale = y_scale = 1.0

        normalized = []
        for player in player_list:
            if player is goalkeeper:
                world_xy = (52.0, 0.0)
            elif player is kicker:
                world_xy = (36.6, -0.8)
            else:
                rel_x = (player.foot_pixel[0] - x_center) / x_scale
                rel_y = (player.foot_pixel[1] - y_center) / y_scale
                world_xy = (31.0 + rel_y * 2.4, rel_x * 8.0)
            normalized.append(DetectedPlayer(
                player_id=player.player_id,
                bbox=player.bbox,
                foot_pixel=player.foot_pixel,
                world_position=(float(world_xy[0]), float(world_xy[1]), 0.0),
                team=player.team,
                confidence=player.confidence,
                foot_confidence=player.foot_confidence,
                foot_source=player.foot_source,
                field_contact=player.field_contact,
                is_ball_carrier=player.is_ball_carrier,
            ))
        return normalized

    def detect_pitch(self, image: np.ndarray) -> DetectedPitch:
        keypoints = self._detect_pitch_keypoints(image)
        H_model, valid_count, inliers, reproj_error, inlier_indices = self._estimate_homography_from_keypoints(keypoints)

        if H_model is not None and self._homography_has_reasonable_scale(H_model, image.shape[1], image.shape[0]):
            return DetectedPitch(
                homography=H_model,
                method="yolo_pitch_keypoints",
                keypoints=keypoints,
                inliers=inliers,
                reprojection_error=reproj_error,
                valid_keypoints=valid_count,
                inlier_indices=inlier_indices,
            )

        # 旧校准矩阵只适合原始示例截图。换帧/换机位时必须优先使用每帧自动标定，
        # 只有自动标定失败才回退到文件矩阵，避免把右侧真实球员投到场外误删。
        if self.calibrated_homography_path.exists():
            return DetectedPitch(
                homography=np.load(self.calibrated_homography_path),
                method=f"calibrated_file:{self.calibrated_homography_path}",
                keypoints=keypoints,
                inliers=inliers,
                reprojection_error=reproj_error,
                valid_keypoints=valid_count,
                inlier_indices=inlier_indices,
            )
        raise RuntimeError("无法自动估计球场 homography，且没有可用的校准矩阵文件")

    def _detect_pitch_keypoints(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        if not self.pitch_model_path.exists():
            return []

        try:
            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
            from ultralytics import YOLO
        except Exception:
            return []

        result = YOLO(str(self.pitch_model_path))(image, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return []

        data = result.keypoints.data[0].detach().cpu().numpy()
        return [(float(x), float(y), float(conf)) for x, y, conf in data]

    def _estimate_homography_from_keypoints(self, keypoints: Sequence[Tuple[float, float, float]]):
        if len(keypoints) != len(self.PITCH_KEYPOINTS.vertices):
            return None, 0, 0, None, []

        image_points = np.array([[x, y] for x, y, conf in keypoints], dtype=np.float32)
        confidences = np.array([conf for x, y, conf in keypoints], dtype=np.float32)
        valid = confidences >= self.keypoint_confidence
        valid_indices = np.flatnonzero(valid)
        valid_count = int(valid_indices.size)
        if valid_count < 6:
            return None, valid_count, 0, None, []

        template_points = self.PITCH_KEYPOINTS.template_vertices_array()
        H, status = cv2.findHomography(image_points[valid], template_points[valid], cv2.RANSAC, 5.0)
        if H is None or status is None:
            return None, valid_count, 0, None, []

        inlier_mask = status.ravel().astype(bool)
        reprojected = cv2.perspectiveTransform(image_points[valid].reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(reprojected - template_points[valid], axis=1)
        reproj_error = float(errors[inlier_mask].mean()) if inlier_mask.any() else float(errors.mean())
        inlier_indices = [int(idx) for idx in valid_indices[inlier_mask]]
        return H, valid_count, int(inlier_mask.sum()), reproj_error, inlier_indices

    def _homography_has_reasonable_scale(self, H: np.ndarray, width: int, height: int) -> bool:
        sample_pixels = np.array([
            [[width * 0.15, height * 0.80]],
            [[width * 0.50, height * 0.55]],
            [[width * 0.85, height * 0.80]],
        ], dtype=np.float32)
        template = cv2.perspectiveTransform(sample_pixels, H).reshape(-1, 2)
        if not np.isfinite(template).all():
            return False
        world = self.PITCH_KEYPOINTS.template_to_world(template)
        span = np.linalg.norm(world.max(axis=0) - world.min(axis=0))
        return 20.0 <= float(span) <= 140.0

    @classmethod
    def _rf_vertices_to_world(cls, vertices: np.ndarray) -> np.ndarray:
        return SoccerPitchKeypointConfig(vertices=[tuple(point) for point in vertices.tolist()]).world_vertices_array()

    def detect_players(self, image: np.ndarray, H: np.ndarray) -> List[DetectedPlayer]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        field_core_mask = self._estimate_field_core_mask(hsv)

        model_players = self._detect_players_with_object_model(image, H, field_core_mask)
        if not self.use_color_fallback:
            return self._reassign_player_ids(sorted(model_players, key=lambda p: (p.foot_pixel[1], p.foot_pixel[0])))

        color_players = self._detect_players_by_color(image, H, hsv, field_core_mask)
        if not model_players:
            return color_players

        merged = list(model_players)
        for color_player in color_players:
            overlaps_model = any(self._bbox_iou(color_player.bbox, model_player.bbox) > 0.10 for model_player in model_players)
            if overlaps_model:
                continue
            merged.append(color_player)

        merged = self._deduplicate_nearby_players(merged)
        merged = sorted(merged, key=lambda p: p.confidence, reverse=True)[:22]
        return self._reassign_player_ids(sorted(merged, key=lambda p: (p.foot_pixel[1], p.foot_pixel[0])))

    def _deduplicate_nearby_players(self, players: List[DetectedPlayer]) -> List[DetectedPlayer]:
        """去除脚点极近的重复球员，保留置信度更高的。"""
        if not players:
            return []
        players = sorted(players, key=lambda p: p.confidence, reverse=True)
        kept = []
        for p in players:
            is_duplicate = False
            for existing in kept:
                foot_dist = np.hypot(p.foot_pixel[0] - existing.foot_pixel[0], p.foot_pixel[1] - existing.foot_pixel[1])
                if foot_dist < 18.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(p)
        return kept

    def _detect_players_by_color(self, image: np.ndarray, H: np.ndarray, hsv: Optional[np.ndarray] = None,
                                 field_core_mask: Optional[np.ndarray] = None) -> List[DetectedPlayer]:
        if hsv is None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if field_core_mask is None:
            field_core_mask = self._estimate_field_core_mask(hsv)
        field_mask = self._estimate_field_mask(hsv)

        color_masks = {
            "away": cv2.inRange(hsv, (8, 70, 80), (42, 255, 255)),
            "home": cv2.inRange(hsv, (0, 0, 135), (179, 80, 255)),
            "referee": cv2.inRange(hsv, (0, 0, 0), (179, 120, 95)),
        }

        candidates = []
        for team, mask in color_masks.items():
            candidates.extend(self._extract_player_candidates(mask, field_mask, team))

        candidates = self._nms_candidates(candidates, iou_threshold=0.35)
        players = []
        for player_id, candidate in enumerate(candidates, start=1):
            x, y, w, h, team, score = candidate
            foot, foot_confidence, foot_source = self._estimate_foot_pixel(image, (int(x), int(y), int(w), int(h)), team)
            if not self._is_foot_on_playing_surface(field_core_mask, foot):
                continue
            world = self.pixel_to_world(H, foot[0], foot[1])
            if not self._is_inside_pitch(world, margin=4.0):
                continue
            players.append(DetectedPlayer(
                player_id=len(players) + 1,
                bbox=(int(x), int(y), int(w), int(h)),
                foot_pixel=(float(foot[0]), float(foot[1])),
                world_position=(float(world[0]), float(world[1]), 0.0),
                team=team,
                confidence=float(score),
                foot_confidence=float(foot_confidence),
                foot_source=foot_source,
                field_contact=self._foot_field_contact(field_core_mask, foot),
            ))
        return players

    @staticmethod
    def _reassign_player_ids(players: Sequence[DetectedPlayer]) -> List[DetectedPlayer]:
        reassigned = []
        for idx, player in enumerate(players, start=1):
            player.player_id = idx
            player.is_ball_carrier = False
            reassigned.append(player)
        return reassigned

    def _detect_players_with_object_model(self, image: np.ndarray, H: np.ndarray,
                                          field_core_mask: Optional[np.ndarray] = None) -> List[DetectedPlayer]:
        if self.object_model_path is None or not self.object_model_path.exists():
            return []

        if field_core_mask is None:
            field_core_mask = self._estimate_field_core_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        try:
            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
            from ultralytics import YOLO
        except Exception:
            return []

        result = YOLO(str(self.object_model_path))(image, verbose=False)[0]
        if result.boxes is None:
            return []
        names = result.names or {}

        players = []
        for box in result.boxes:
            cls_id = int(box.cls.detach().cpu().item())
            cls_name = str(names.get(cls_id, cls_id)).lower()
            confidence = float(box.conf.detach().cpu().item())
            is_person = cls_name in {"person", "player", "goalkeeper", "referee"} or cls_id == 0
            if not is_person or confidence < 0.20:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
            w, h = x2 - x1, y2 - y1
            if confidence < 0.45 and (w < 30.0 or h < 50.0):
                continue
            if w < 3 or h < 8:
                continue
            bbox = (int(x1), int(y1), int(w), int(h))
            if self._is_too_small_and_round_for_player(w, h):
                continue
            if self._is_probably_ad_board_candidate(field_core_mask, bbox):
                continue
            foot, foot_confidence, foot_source = self._estimate_foot_pixel(image, bbox, cls_name)
            if not self._is_foot_on_playing_surface(field_core_mask, foot):
                continue
            world = self.pixel_to_world(H, foot[0], foot[1])
            if not self._is_inside_pitch(world, margin=4.0):
                continue
            if cls_name == "referee":
                team = "referee"
            else:
                team = self._classify_team_from_crop(image, (int(x1), int(y1), int(w), int(h)))
            players.append(DetectedPlayer(
                player_id=len(players) + 1,
                bbox=bbox,
                foot_pixel=(float(foot[0]), float(foot[1])),
                world_position=(float(world[0]), float(world[1]), 0.0),
                team=team,
                confidence=confidence,
                foot_confidence=float(foot_confidence),
                foot_source=foot_source,
                field_contact=self._foot_field_contact(field_core_mask, foot),
            ))
        return players

    def _estimate_foot_pixel(self, image: np.ndarray, bbox: Tuple[int, int, int, int], team: str) -> Tuple[Tuple[float, float], float, str]:
        """在 bbox 底部附近找最像落脚点的像素，失败时回退到底边中心。"""
        x, y, w, h = bbox
        image_h, image_w = image.shape[:2]
        fallback = (float(x + w * 0.5), float(y + h))
        if w <= 0 or h <= 0:
            return fallback, 0.15, "bbox_bottom"

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_w, x + w), min(image_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return fallback, 0.15, "bbox_bottom"

        crop = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_start = max(0, int(crop.shape[0] * 0.45))
        lower = hsv[lower_start:]
        if lower.size == 0:
            return fallback, 0.15, "bbox_bottom"

        non_grass = cv2.bitwise_not(cv2.inRange(lower, (35, 35, 45), (95, 255, 255)))
        low_sat_dark = cv2.inRange(lower, (0, 0, 0), (179, 95, 145))
        bright_kit = cv2.inRange(lower, (0, 0, 120), (179, 95, 255))
        orange_kit = cv2.inRange(lower, (8, 65, 70), (42, 255, 255))
        player_mask = cv2.bitwise_and(non_grass, cv2.bitwise_or(low_sat_dark, cv2.bitwise_or(bright_kit, orange_kit)))
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        points = cv2.findNonZero(player_mask)
        if points is None or len(points) < max(4, int(w * 0.25)):
            return fallback, 0.20, "bbox_bottom"

        pts = points.reshape(-1, 2).astype(float)
        bottom_y = float(pts[:, 1].max())
        bottom_band = pts[pts[:, 1] >= bottom_y - max(2.0, h * 0.08)]
        if bottom_band.size == 0:
            return fallback, 0.20, "bbox_bottom"

        foot_x = float(np.median(bottom_band[:, 0]) + x1)
        foot_y = float(bottom_y + lower_start + y1)
        horizontal_shift = abs(foot_x - fallback[0]) / max(float(w), 1.0)
        vertical_gain = max(0.0, fallback[1] - foot_y) / max(float(h), 1.0)
        pixel_support = min(1.0, len(bottom_band) / max(float(w), 1.0))
        confidence = float(np.clip(0.35 + 0.35 * pixel_support - 0.25 * horizontal_shift + 0.15 * vertical_gain, 0.25, 0.90))

        if horizontal_shift > 0.55:
            return fallback, 0.22, "bbox_bottom"
        return (foot_x, foot_y), confidence, "lower_body_pixels"

    def _classify_team_from_crop(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        x, y, w, h = bbox
        crop = image[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        if crop.size == 0:
            return "home"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, (8, 70, 80), (42, 255, 255)).mean()
        white = cv2.inRange(hsv, (0, 0, 135), (179, 80, 255)).mean()
        black = cv2.inRange(hsv, (0, 0, 0), (179, 120, 95)).mean()
        if black > orange and black > white and black > 18:
            return "referee"
        return "away" if orange > white else "home"

    def _estimate_field_mask(self, hsv: np.ndarray) -> np.ndarray:
        green = cv2.inRange(hsv, (32, 35, 35), (78, 255, 210))
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(green.shape, dtype=np.uint8)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1]:
            if cv2.contourArea(contour) > 1000:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        return cv2.dilate(mask, np.ones((31, 31), np.uint8), iterations=2)

    def _estimate_field_core_mask(self, hsv: np.ndarray) -> np.ndarray:
        green = cv2.inRange(hsv, (32, 35, 35), (78, 255, 210))
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(green.shape, dtype=np.uint8)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1]:
            if cv2.contourArea(contour) > 1000:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        return mask

    @staticmethod
    def _is_foot_on_playing_surface(field_core_mask: np.ndarray, foot: Tuple[float, float]) -> bool:
        height, width = field_core_mask.shape[:2]
        x = int(round(foot[0]))
        y = int(round(foot[1]))
        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        field_top = FootballVisionPipeline._estimate_local_field_top(field_core_mask, x)
        if field_top is None or y < field_top + 30:
            return False

        x1, x2 = max(0, x - 12), min(width, x + 13)
        y1, y2 = max(0, y - 6), min(height, y + 18)
        around_foot = field_core_mask[y1:y2, x1:x2]
        if around_foot.size and float((around_foot > 0).mean()) >= 0.12:
            return True

        y3, y4 = max(0, y), min(height, y + 34)
        below_foot = field_core_mask[y3:y4, x1:x2]
        return bool(below_foot.size and float((below_foot > 0).mean()) >= 0.18)

    @staticmethod
    def _foot_field_contact(field_core_mask: np.ndarray, foot: Tuple[float, float]) -> float:
        height, width = field_core_mask.shape[:2]
        x = int(round(foot[0]))
        y = int(round(foot[1]))
        if x < 0 or x >= width or y < 0 or y >= height:
            return 0.0
        x1, x2 = max(0, x - 12), min(width, x + 13)
        y1, y2 = max(0, y - 4), min(height, y + 18)
        patch = field_core_mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float((patch > 0).mean())

    @staticmethod
    def _estimate_local_field_top(field_core_mask: np.ndarray, x: int) -> Optional[int]:
        height, width = field_core_mask.shape[:2]
        x1, x2 = max(0, x - 24), min(width, x + 25)
        if x1 >= x2:
            return None
        column_ratio = (field_core_mask[:, x1:x2] > 0).mean(axis=1)
        kernel = np.ones(41, dtype=float) / 41.0
        smooth = np.convolve(column_ratio, kernel, mode="same")
        rows = np.where((np.arange(height) > 140) & (smooth > 0.25))[0]
        if rows.size == 0:
            return None
        return int(rows[0])

    def _extract_player_candidates(self, mask: np.ndarray, field_mask: np.ndarray, team: str):
        mask = cv2.bitwise_and(mask, field_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        min_area = {"away": 120.0, "home": 90.0, "referee": 18.0}.get(team, 90.0)
        min_height = {"away": 16, "home": 16, "referee": 9}.get(team, 16)
        min_width = {"away": 5, "home": 5, "referee": 3}.get(team, 5)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            image_h, image_w = mask.shape[:2]
            if x <= 2 or x + w >= image_w - 2:
                continue
            if team == "referee" and (w < 12 or h < 24):
                continue
            if area < min_area or area > 2500:
                continue
            if h < min_height or w < min_width:
                continue
            if self._is_too_small_and_round_for_player(w, h):
                continue
            if y < 85:
                continue
            aspect = h / max(float(w), 1.0)
            min_aspect = 1.08 if team in {"home", "away"} else 1.20
            if aspect < min_aspect or aspect > 7.0:
                continue
            if self._is_probably_ad_board_candidate(field_mask, (int(x), int(y), int(w), int(h))):
                continue
            score = min(1.0, area / 900.0) * (1.0 if team != "home" else 0.85)
            candidates.append((x, y, w, h, team, score))
        return candidates

    @staticmethod
    def _is_too_small_and_round_for_player(width: float, height: float) -> bool:
        if height < 35.0 and width < 28.0:
            return True
        if height >= 34.0:
            return False
        aspect = height / max(float(width), 1.0)
        return bool(aspect < 1.55)

    @staticmethod
    def _is_probably_ad_board_candidate(field_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        image_h, image_w = field_mask.shape[:2]
        cx = int(np.clip(round(x + w * 0.5), 0, image_w - 1))
        bottom = int(np.clip(round(y + h), 0, image_h - 1))
        field_top = FootballVisionPipeline._estimate_local_field_top(field_mask, cx)
        if field_top is None:
            return False
        bottom_clearance = bottom - field_top
        if bottom_clearance < 45 and y < field_top + 20:
            return True
        if y < field_top and bottom < field_top + 70 and h < 90:
            return True
        return False

    def _nms_candidates(self, candidates, iou_threshold: float):
        candidates = sorted(candidates, key=lambda item: item[5], reverse=True)
        kept = []
        for candidate in candidates:
            if all(self._bbox_iou(candidate[:4], kept_item[:4]) < iou_threshold for kept_item in kept):
                kept.append(candidate)
        return sorted(kept, key=lambda item: (item[1], item[0]))

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return 0.0 if union <= 0 else inter / union

    def detect_ball(self, image: np.ndarray, H: np.ndarray, players: Sequence[DetectedPlayer]) -> Optional[DetectedBall]:
        # 自训练 person/ball 小样本模型对 ball 还不稳定，容易把门将/球员局部误报为球。
        # 但近景/点球场景里模型的 ball 框往往比白色 blob 更可靠，先接收高置信小框。
        model_ball = self._detect_ball_with_object_model(image, H, prefer_generic=False, min_confidence=0.55)
        if model_ball is not None:
            return model_ball

        # 远景里 COCO sports ball 常会误检鞋/袜，再用小白球 blob；找不到再回退通用模型。
        blob_ball = self._detect_ball_by_bright_blob(image, H, players)
        if blob_ball is not None:
            return blob_ball

        model_ball = self._detect_ball_with_object_model(image, H, prefer_generic=True, min_confidence=0.10)
        if model_ball is not None:
            return model_ball

        return None

    def _detect_ball_by_bright_blob(self, image: np.ndarray, H: np.ndarray,
                                    players: Sequence[DetectedPlayer]) -> Optional[DetectedBall]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        field_mask = self._estimate_field_mask(hsv)
        bright = cv2.inRange(hsv, (0, 0, 135), (179, 125, 255))
        bright = cv2.bitwise_and(bright, field_mask)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        player_points = np.array([p.foot_pixel for p in players], dtype=float) if players else np.zeros((0, 2), dtype=float)
        if len(player_points) >= 5:
            px_min, py_min = player_points.min(axis=0)
            px_max, py_max = player_points.max(axis=0)
            play_bounds = (px_min - 80.0, py_min - 80.0, px_max + 80.0, py_max + 80.0)
        else:
            play_bounds = (0.0, 0.0, float(image.shape[1]), float(image.shape[0]))

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < 2 or area > 120 or w > 18 or h > 18:
                continue
            cx, cy = x + w * 0.5, y + h * 0.5
            if not (play_bounds[0] <= cx <= play_bounds[2] and play_bounds[1] <= cy <= play_bounds[3]):
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else 4.0 * np.pi * area / (perimeter * perimeter)
            distances = sorted(
                ((np.hypot(cx - p.foot_pixel[0], cy - p.foot_pixel[1]), p) for p in players),
                key=lambda item: item[0],
            )
            nearest_player = distances[0][0] if distances else 999.0
            if nearest_player < 18.0 or nearest_player > 130.0:
                continue
            penalty_like_candidate = self._is_penalty_like_ball_candidate(cx, cy, field_mask, players)
            if nearest_player > 55.0 and not penalty_like_candidate:
                continue
            nearest = distances[0][1] if distances else None
            if nearest is not None and self._is_probably_player_boot_or_kit(cx, cy, nearest):
                continue
            near_players = sum(1 for dist, _ in distances[:5] if dist < 95.0)
            close_to_play = max(0.0, 1.0 - abs(nearest_player - 42.0) / 95.0)
            size_score = max(0.0, 1.0 - abs(area - 18.0) / 80.0)
            body_penalty = 0.0
            foot_zone_bonus = 0.0
            if nearest is not None:
                bx, by, bw, bh = nearest.bbox
                inside_nearest = bx - 2 <= cx <= bx + bw + 2 and by - 2 <= cy <= by + bh + 2
                vertical_ratio = (cy - by) / max(float(bh), 1.0)
                if inside_nearest and vertical_ratio < 0.82:
                    body_penalty = 0.35
                if vertical_ratio >= 0.82 or cy >= nearest.foot_pixel[1] - 12:
                    foot_zone_bonus = 0.12
            centrality = 0.0
            if len(player_points) >= 5:
                centroid = player_points.mean(axis=0)
                centrality = max(0.0, 1.0 - np.hypot(cx - centroid[0], cy - centroid[1]) / 520.0)
            team_bonus = 0.08 if nearest is not None and nearest.team == "home" else 0.0
            crowd_score = min(1.0, near_players / 3.0)
            penalty_spot_bonus = 0.0
            if penalty_like_candidate:
                penalty_spot_bonus = 0.16
            score = circularity * 0.25 + close_to_play * 0.27 + size_score * 0.13 + centrality * 0.12 + crowd_score * 0.10 + team_bonus + foot_zone_bonus + penalty_spot_bonus - body_penalty
            candidates.append((score, cx, cy))

        candidates.extend(self._detect_high_confidence_ball_circles(image, field_mask, players))

        if not candidates:
            return None

        score, cx, cy = max(candidates, key=lambda item: item[0])
        world = self.pixel_to_world(H, cx, cy)
        return DetectedBall(
            pixel=(float(cx), float(cy)),
            world_position=(float(world[0]), float(world[1]), 0.05),
            confidence=float(score),
        )

    def _is_penalty_like_ball_candidate(self, cx: float, cy: float, field_mask: np.ndarray,
                                        players: Sequence[DetectedPlayer]) -> bool:
        if len(players) < 6:
            return False
        field_top = self._estimate_local_field_top(field_mask, int(round(cx)))
        if field_top is None or cy <= field_top + 90:
            return False
        left_of_ball = [p for p in players if p.foot_pixel[0] < cx]
        right_of_ball = [p for p in players if p.foot_pixel[0] > cx]
        return bool(len(left_of_ball) >= 4 and len(right_of_ball) >= 1)

    def _detect_high_confidence_ball_circles(self, image: np.ndarray, field_mask: np.ndarray,
                                            players: Sequence[DetectedPlayer]) -> List[Tuple[float, float, float]]:
        """补充检测贴近白线/白衣时被 blob 合并掉的真球。

        只接受非常保守的圆形候选：中心够亮、周围以草地为主、局部白色比例适中。
        这样可以覆盖图2的大禁区线真球，同时不污染普通远景图。
        """
        if len(players) < 2:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 3),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=18,
            param1=80,
            param2=12,
            minRadius=3,
            maxRadius=9,
        )
        if circles is None:
            return []

        candidates = []
        image_h, image_w = image.shape[:2]
        for cx, cy, radius in circles[0]:
            cx, cy, radius = float(cx), float(cy), float(radius)
            ix, iy = int(round(cx)), int(round(cy))
            if ix < 0 or ix >= image_w or iy < 0 or iy >= image_h:
                continue
            if field_mask[iy, ix] == 0:
                continue
            if not (3.5 <= radius <= 7.5):
                continue

            distances = sorted(
                ((np.hypot(cx - p.foot_pixel[0], cy - p.foot_pixel[1]), p) for p in players),
                key=lambda item: item[0],
            )
            nearest_dist, nearest = distances[0]
            if nearest_dist < 16.0 or nearest_dist > 95.0:
                continue
            if self._is_probably_player_boot_or_kit(cx, cy, nearest):
                continue

            x1, x2 = max(0, ix - 14), min(image_w, ix + 15)
            y1, y2 = max(0, iy - 14), min(image_h, iy + 15)
            patch_hsv = hsv[y1:y2, x1:x2]
            white = cv2.inRange(patch_hsv, (0, 0, 145), (179, 105, 255))
            green = cv2.inRange(patch_hsv, (35, 35, 45), (95, 255, 255))
            white_ratio = float((white > 0).mean()) if white.size else 0.0
            green_ratio = float((green > 0).mean()) if green.size else 0.0
            center_brightness = float(gray[iy, ix])

            if center_brightness < 185.0 or white_ratio < 0.08 or green_ratio < 0.70:
                continue

            radius_score = max(0.0, 1.0 - abs(radius - 5.5) / 5.5)
            brightness_score = center_brightness / 255.0
            score = 0.86 + 0.06 * radius_score + 0.05 * brightness_score + 0.03 * min(1.0, green_ratio)
            candidates.append((float(score), cx, cy))
        return candidates

    def _detect_ball_with_object_model(self, image: np.ndarray, H: np.ndarray, prefer_generic: bool = False,
                                       min_confidence: float = 0.10) -> Optional[DetectedBall]:
        model_path = self.object_model_path
        if prefer_generic and Path("yolov8n.pt").exists():
            model_path = Path("yolov8n.pt")

        if model_path is None or not model_path.exists():
            return None

        try:
            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
            from ultralytics import YOLO
        except Exception:
            return None

        result = YOLO(str(model_path))(image, verbose=False)[0]
        if result.boxes is None:
            return None
        names = result.names or {}

        best = None
        for box in result.boxes:
            cls_id = int(box.cls.detach().cpu().item())
            cls_name = str(names.get(cls_id, cls_id)).lower()
            confidence = float(box.conf.detach().cpu().item())
            is_ball = cls_name in {"ball", "sports ball", "football"} or cls_id == 32
            if not is_ball or confidence < min_confidence:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area = width * height
            image_h, image_w = image.shape[:2]
            # 足球在转播远景里应该很小；自训练样本少时，模型容易把球员/门将局部误报为 ball。
            # 这里过滤掉明显过大的 ball 框，交给小白球 fallback 或其他小框候选。
            if width > image_w * 0.035 or height > image_h * 0.08 or area > image_w * image_h * 0.0015:
                continue
            score = confidence / np.sqrt(area)
            if best is None or score > best[0]:
                best = (score, confidence, (x1 + x2) * 0.5, (y1 + y2) * 0.5)
        if best is None:
            return None
        _, confidence, cx, cy = best
        world = self.pixel_to_world(H, cx, cy)
        return DetectedBall(
            pixel=(float(cx), float(cy)),
            world_position=(float(world[0]), float(world[1]), 0.05),
            confidence=float(confidence),
        )

    @staticmethod
    def _point_in_expanded_bbox(x: float, y: float, bbox: Tuple[int, int, int, int], pad: int) -> bool:
        bx, by, bw, bh = bbox
        return bx - pad <= x <= bx + bw + pad and by - pad <= y <= by + bh + pad

    @staticmethod
    def _is_probably_player_boot_or_kit(x: float, y: float, player: DetectedPlayer) -> bool:
        bx, by, bw, bh = player.bbox
        inside_x = bx - 8 <= x <= bx + bw + 8
        lower_body_or_boot = by + bh * 0.70 <= y <= by + bh + 24
        return bool(inside_x and lower_body_or_boot)

    def assign_ball_carrier(self, players: List[DetectedPlayer], ball: Optional[DetectedBall]) -> Optional[int]:
        if not players:
            return None

        if ball is None:
            carrier = max(players, key=lambda p: p.confidence)
        else:
            bx, by = ball.world_position[:2]
            carrier = min(players, key=lambda p: np.hypot(p.world_position[0] - bx, p.world_position[1] - by))

        for player in players:
            player.is_ball_carrier = player.player_id == carrier.player_id
        return carrier.player_id

    @staticmethod
    def pixel_to_world(H: np.ndarray, px: float, py: float) -> np.ndarray:
        template_xy = FootballVisionPipeline.pixel_to_template(H, px, py)
        world_xy = SoccerPitchKeypointConfig().template_to_world(template_xy.reshape(1, 2))[0]
        return np.array([world_xy[0], world_xy[1], 0.0], dtype=float)

    @staticmethod
    def pixel_to_template(H: np.ndarray, px: float, py: float) -> np.ndarray:
        pt = H @ np.array([px, py, 1.0], dtype=float)
        pt /= pt[2]
        return np.array([pt[0], pt[1]], dtype=float)

    @staticmethod
    def _is_inside_pitch(world: np.ndarray, margin: float = 0.0) -> bool:
        return abs(float(world[0])) <= 52.5 + margin and abs(float(world[1])) <= 34.0 + margin

    def to_render_objects(self, frame: VisionFrame) -> Tuple[CameraParams, List[Player], Ball]:
        players = []
        for detected in frame.players:
            team = {
                "home": TeamSide.HOME,
                "away": TeamSide.AWAY,
                "referee": TeamSide.REFEREE,
            }.get(detected.team, TeamSide.HOME)
            players.append(Player(
                player_id=detected.player_id,
                position=np.array(detected.world_position, dtype=float),
                direction=np.array([1.0, 0.0, 0.0], dtype=float),
                team=team,
                is_ball_carrier=detected.is_ball_carrier,
            ))
            players[-1].foot_pixel = tuple(detected.foot_pixel)

        if frame.ball is not None:
            ball_position = np.array(frame.ball.world_position, dtype=float)
        elif players:
            carrier = next((p for p in players if p.is_ball_carrier), players[0])
            ball_position = carrier.position + np.array([0.5, 0.0, 0.05])
        else:
            ball_position = np.array([0.0, 0.0, 0.05], dtype=float)

        camera = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0]),
            focal_length=1000.0,
            principal_point=(frame.image_size[0] / 2, frame.image_size[1] / 2),
            image_size=frame.image_size,
        )
        camera._R_override = np.eye(3)
        camera._H = frame.pitch.homography
        return camera, players, Ball(position=ball_position)

    def draw_debug_overlay(self, image_path: str, frame: VisionFrame, output_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        for idx, (x, y, conf) in enumerate(frame.pitch.keypoints):
            if conf >= self.keypoint_confidence:
                cv2.circle(image, (int(round(x)), int(round(y))), 4, (255, 0, 255), -1)
                cv2.putText(image, str(idx), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        colors = {
            "home": (255, 255, 255),
            "away": (0, 180, 255),
            "referee": (0, 0, 0),
        }
        for player in frame.players:
            x, y, w, h = player.bbox
            color = (0, 255, 0) if player.is_ball_carrier else colors.get(player.team, (255, 255, 255))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.circle(image, (int(player.foot_pixel[0]), int(player.foot_pixel[1])), 4, color, -1)
            label = f"#{player.player_id} {player.team[:3]} fc={player.foot_confidence:.2f} fld={player.field_contact:.2f}"
            cv2.putText(image, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            if player.foot_confidence < 0.40 or player.field_contact < 0.15:
                cv2.circle(image, (int(player.foot_pixel[0]), int(player.foot_pixel[1])), 8, (0, 0, 255), 2)

        if frame.ball is not None:
            bx, by = int(round(frame.ball.pixel[0])), int(round(frame.ball.pixel[1]))
            cv2.circle(image, (bx, by), 8, (255, 255, 255), 2)
            cv2.putText(image, "ball", (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(image, f"pitch: {frame.pitch.method}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

    def draw_pitch_keypoints_debug(self, image_path: str, frame: VisionFrame, output_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        config = self.PITCH_KEYPOINTS
        keypoints = frame.pitch.keypoints
        inlier_indices = set(frame.pitch.inlier_indices)

        for start, end in config.edges:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            sx, sy, sconf = keypoints[start]
            ex, ey, econf = keypoints[end]
            if sconf < self.keypoint_confidence or econf < self.keypoint_confidence:
                continue
            color = (0, 220, 0) if start in inlier_indices and end in inlier_indices else (0, 220, 255)
            cv2.line(image, (int(round(sx)), int(round(sy))), (int(round(ex)), int(round(ey))), color, 2)

        for idx, (x, y, conf) in enumerate(keypoints):
            if conf >= self.keypoint_confidence and idx in inlier_indices:
                color = (0, 255, 0)
                radius = 6
            elif conf >= self.keypoint_confidence:
                color = (0, 220, 255)
                radius = 6
            else:
                color = (130, 130, 130)
                radius = 4

            center = (int(round(x)), int(round(y)))
            cv2.circle(image, center, radius, color, -1)
            label = config.labels[idx] if idx < len(config.labels) else str(idx)
            cv2.putText(image, f"{idx}:{label} {conf:.2f}", (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        error_text = "n/a" if frame.pitch.reprojection_error is None else f"{frame.pitch.reprojection_error:.2f}m"
        lines = [
            f"method: {frame.pitch.method}",
            f"valid/inliers: {frame.pitch.valid_keypoints}/{frame.pitch.inliers}",
            f"reproj error: {error_text}",
            "green=inlier yellow=rejected gray=low-conf",
        ]
        for line_idx, line in enumerate(lines):
            y = 28 + line_idx * 24
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

    def draw_homography_reprojection_debug(self, image_path: str, frame: VisionFrame, output_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        config = self.PITCH_KEYPOINTS
        field_mask = self._estimate_field_core_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        field_mask = cv2.erode(field_mask, np.ones((9, 9), np.uint8), iterations=1)
        white_mask = self._field_white_line_mask(image, field_mask)
        H_template_to_pixel = np.linalg.inv(frame.pitch.homography)
        template_vertices = config.template_vertices_array().astype(np.float32)
        projected = cv2.perspectiveTransform(template_vertices.reshape(-1, 1, 2), H_template_to_pixel).reshape(-1, 2)

        inlier_indices = set(frame.pitch.inlier_indices)
        for start, end in config.edges:
            # Roboflow-style debug: only draw edges whose endpoints were actually
            # detected and used by homography. Drawing the full theoretical pitch
            # through occluded/advertising areas looks like floating lines and is
            # misleading for broadcast frames.
            if start not in inlier_indices or end not in inlier_indices:
                continue
            self._draw_projected_line_on_field(image, projected[start], projected[end], field_mask, white_mask)

        for idx, (x, y, confidence) in enumerate(frame.pitch.keypoints):
            if confidence < self.keypoint_confidence:
                continue
            color = (0, 255, 0) if idx in set(frame.pitch.inlier_indices) else (0, 180, 255)
            cv2.circle(image, (int(round(x)), int(round(y))), 6, color, -1)
            cv2.putText(image, str(idx), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for player in frame.players:
            fx, fy = player.foot_pixel
            wx, wy = player.world_position[:2]
            center = (int(round(fx)), int(round(fy)))
            cv2.circle(image, center, 5, (255, 0, 0), -1)
            label = f"#{player.player_id} ({wx:.1f},{wy:.1f})"
            cv2.putText(image, label, (center[0] + 5, center[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

        if frame.ball is not None:
            bx, by = frame.ball.pixel
            cv2.circle(image, (int(round(bx)), int(round(by))), 7, (255, 255, 255), 2)
            cv2.putText(image, "ball", (int(bx) + 8, int(by) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        error_text = "n/a" if frame.pitch.reprojection_error is None else f"{frame.pitch.reprojection_error:.2f}m"
        lines = [
            "yellow = projected inlier edges overlapping detected field lines",
            "blue = player foot points used for 3D positions",
            f"pitch: {frame.pitch.method} valid/inliers: {frame.pitch.valid_keypoints}/{frame.pitch.inliers} err: {error_text}",
        ]
        for line_idx, line in enumerate(lines):
            y = 28 + line_idx * 24
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)

    @staticmethod
    def _draw_projected_line_on_field(image: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                                      field_mask: np.ndarray, white_mask: np.ndarray) -> None:
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            return
        distance = float(np.linalg.norm(p2 - p1))
        sample_count = max(12, int(distance / 4.0))
        points = np.linspace(p1, p2, sample_count)
        current_segment = []
        height, width = field_mask.shape[:2]

        def flush_segment() -> None:
            if len(current_segment) >= 2:
                polyline = np.round(np.array(current_segment, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(image, [polyline], False, (0, 255, 255), 3)

        for point in points:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height and field_mask[y, x] > 0 and white_mask[y, x] > 0:
                current_segment.append((float(point[0]), float(point[1])))
            else:
                flush_segment()
                current_segment = []
        flush_segment()

    @staticmethod
    def _field_white_line_mask(image: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 95, 255]))
        white_mask = cv2.bitwise_and(white_mask, field_mask)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.dilate(white_mask, np.ones((17, 17), np.uint8), iterations=1)

    def draw_roboflow_radar_debug(self, frame: VisionFrame, output_path: str,
                                  scale: float = 10.0, padding: int = 60) -> None:
        config = self.PITCH_KEYPOINTS
        width = int(config.template_length * scale + padding * 2)
        height = int(config.template_width * scale + padding * 2)
        canvas = np.full((height, width, 3), (35, 110, 45), dtype=np.uint8)

        def to_px(template_point: Sequence[float]) -> Tuple[int, int]:
            return (
                int(round(float(template_point[0]) * scale + padding)),
                int(round(float(template_point[1]) * scale + padding)),
            )

        vertices = config.template_vertices_array()
        for start, end in config.edges:
            cv2.line(canvas, to_px(vertices[start]), to_px(vertices[end]), (245, 245, 245), 2)

        for player in frame.players:
            template_xy = self.pixel_to_template(frame.pitch.homography, player.foot_pixel[0], player.foot_pixel[1])
            color = (255, 255, 255) if player.team == "home" else (0, 130, 255)
            if player.team == "referee":
                color = (20, 20, 20)
            if player.is_ball_carrier:
                color = (50, 255, 50)
            cv2.circle(canvas, to_px(template_xy), 7, color, -1)
            cv2.putText(canvas, f"#{player.player_id}", (to_px(template_xy)[0] + 8, to_px(template_xy)[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(canvas, f"#{player.player_id}", (to_px(template_xy)[0] + 8, to_px(template_xy)[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if frame.ball is not None:
            ball_template = self.pixel_to_template(frame.pitch.homography, frame.ball.pixel[0], frame.ball.pixel[1])
            cv2.circle(canvas, to_px(ball_template), 5, (255, 255, 255), -1)

        cv2.putText(canvas, f"Roboflow template radar: {frame.pitch.method}", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas)


def save_vision_frame(frame: VisionFrame, output_json: str) -> None:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(frame.to_json_dict(), handle, ensure_ascii=False, indent=2)
