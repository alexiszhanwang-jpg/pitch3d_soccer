"""通用足球场景图导出。

这个模块是视觉识别和渲染引擎之间的稳定中间层。
Python 侧只负责产出结构化场景；Three.js / Unity / Godot 等引擎只消费 JSON。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from src.pitch_3d import PitchDimensions
from src.player_renderer import Ball, Player, TeamSide


TEAM_STYLE = {
    TeamSide.HOME.value: {
        "name": "argentina",
        "kit_color": "#b9e6ff",
        "accent_color": "#ffffff",
    },
    TeamSide.AWAY.value: {
        "name": "netherlands",
        "kit_color": "#ff7a1a",
        "accent_color": "#111111",
    },
    TeamSide.REFEREE.value: {
        "name": "referee",
        "kit_color": "#111111",
        "accent_color": "#f5d000",
    },
}


@dataclass(frozen=True)
class EngineVector:
    """同一世界点在不同引擎里的坐标约定。"""

    world: list[float]
    three: list[float]
    unity: list[float]


def _float(value: Any) -> float:
    return float(np.asarray(value).item())


def _team_value(team: TeamSide | str) -> str:
    if isinstance(team, TeamSide):
        return team.value
    return str(team)


def _position3(position: Sequence[float], y_offset: float = 0.0) -> EngineVector:
    x = _float(position[0])
    pitch_y = _float(position[1])
    z = _float(position[2]) if len(position) > 2 else 0.0
    world = [x, pitch_y, z]
    # 项目世界: X=球场长度, Y=球场宽度, Z=高度。
    # Three.js: X=长度, Y=高度, Z=宽度；为了屏幕右手感，宽度取负。
    three = [x, z + y_offset, -pitch_y]
    # Unity: X=长度, Y=高度, Z=宽度。
    unity = [x, z + y_offset, pitch_y]
    return EngineVector(world=world, three=three, unity=unity)


def _direction2(direction: Sequence[float]) -> Dict[str, list[float]]:
    dx = _float(direction[0])
    dy = _float(direction[1])
    norm = float(np.hypot(dx, dy)) or 1.0
    dx /= norm
    dy /= norm
    return {
        "world": [dx, dy, 0.0],
        "three": [dx, 0.0, -dy],
        "unity": [dx, 0.0, dy],
    }


def _normalize_xy(direction: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    vec = np.array([_float(direction[0]), _float(direction[1])], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm >= 1e-6:
        return vec / norm
    fallback_vec = np.array([_float(fallback[0]), _float(fallback[1])], dtype=float)
    fallback_norm = float(np.linalg.norm(fallback_vec))
    if fallback_norm >= 1e-6:
        return fallback_vec / fallback_norm
    return np.array([1.0, 0.0], dtype=float)


def _carrier_view_direction(carrier: Player, ball: Optional[Ball], play_direction: Sequence[float]) -> tuple[np.ndarray, str]:
    """估计持球者真实视角方向。

    俯视战术图适合使用全局进攻/展示方向；第一人称更应该看向脚下球或带球方向。
    当球距离持球者合理时，优先用 carrier -> ball 的方向，避免 FPV 相机偏向整体队形中心。
    """
    fallback = _normalize_xy(play_direction, [1.0, 0.0])
    if ball is None:
        return fallback, "play_direction"

    carrier_xy = np.asarray(carrier.position[:2], dtype=float)
    ball_xy = np.asarray(ball.position[:2], dtype=float)
    ball_vec = ball_xy - carrier_xy
    ball_dist = float(np.linalg.norm(ball_vec))
    if 0.15 <= ball_dist <= 4.0:
        # 近距离球更像“脚下目标”，不是人的视线方向。
        # 尤其图2这种多人争抢球，如果用 carrier->ball，会把 FPV 转向侧后方，视觉上像门将/错误机位。
        return fallback, "play_direction_near_ball"
    if 4.0 < ball_dist <= 12.0:
        ball_dir = ball_vec / ball_dist
        blended = fallback * 0.75 + ball_dir * 0.25
        return _normalize_xy(blended, fallback), "blended_ball"
    if 0.15 <= ball_dist <= 4.0:
        return fallback, "play_direction_near_ball"
    return fallback, "play_direction"


def _maybe_tuple(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    return [float(v) for v in value]


def build_scene_graph(
    *,
    image_path: str,
    image_size: tuple[int, int],
    players: Iterable[Player],
    ball: Optional[Ball],
    carrier: Player,
    play_direction: Sequence[float],
    pitch_method: str = "unknown",
    homography: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """把当前识别结果导出为引擎无关的 scene graph。"""
    dimensions = PitchDimensions()
    direction = _direction2(play_direction)
    view_direction_xy, view_direction_source = _carrier_view_direction(carrier, ball, play_direction)
    view_direction = _direction2([view_direction_xy[0], view_direction_xy[1]])

    carrier_position = _position3(carrier.position, y_offset=1.65)
    look_at = np.asarray(carrier.position, dtype=float).copy()
    look_at[:2] += view_direction_xy * 14.0
    look_at[2] = 1.55
    follow_pos = np.asarray(carrier.position, dtype=float).copy()
    follow_pos[:2] -= view_direction_xy * 8.0
    follow_pos[2] = 5.0

    scene_players = []
    for player in players:
        team = _team_value(player.team)
        style = TEAM_STYLE.get(team, TEAM_STYLE[TeamSide.HOME.value])
        player_direction = getattr(player, "direction", play_direction)
        position = _position3(player.position)
        bbox = getattr(player, "bbox", None)
        foot_pixel = getattr(player, "foot_pixel", None)
        confidence = getattr(player, "confidence", None)
        scene_players.append({
            "id": int(player.player_id),
            "team": team,
            "team_name": style["name"],
            "role": "carrier" if bool(player.is_ball_carrier) else "player",
            "is_carrier": bool(player.is_ball_carrier),
            "position": position.world,
            "engines": {
                "three": {"position": position.three},
                "unity": {"position": position.unity},
            },
            "direction": _direction2(player_direction),
            "height_m": float(getattr(player, "height", 1.8)),
            "radius_m": 0.32,
            "kit_color": style["kit_color"],
            "accent_color": style["accent_color"],
            "foot_pixel": _maybe_tuple(foot_pixel),
            "bbox": _maybe_tuple(bbox),
            "confidence": None if confidence is None else float(confidence),
        })

    ball_node = None
    if ball is not None:
        ball_position = _position3(ball.position)
        ball_node = {
            "position": ball_position.world,
            "engines": {
                "three": {"position": ball_position.three},
                "unity": {"position": ball_position.unity},
            },
            "radius_m": float(getattr(ball, "radius", 0.11)),
        }

    return {
        "schema_version": "0.2",
        "source": {
            "image_path": image_path,
            "image_size": [int(image_size[0]), int(image_size[1])],
        },
        "pitch": {
            "length_m": float(dimensions.length),
            "width_m": float(dimensions.width),
            "method": pitch_method,
            "homography_pixel_to_world": None if homography is None else np.asarray(homography).tolist(),
        },
        "teams": TEAM_STYLE,
        "carrier_id": int(carrier.player_id),
        "play_direction": direction,
        "carrier_view_direction": view_direction,
        "players": scene_players,
        "ball": ball_node,
        "cameras": {
            "tactical": {
                "type": "orbit",
                "fov_degrees": 58.0,
                "target": [0.0, 0.0, 0.0],
                "three_position": [0.0, 70.0, 0.0],
                "unity_position": [0.0, 70.0, 0.0],
            },
            "carrier_follow": {
                "type": "follow",
                "target_player_id": int(carrier.player_id),
                "direction_source": view_direction_source,
                "fov_degrees": 64.0,
                "three_position": _position3(follow_pos).three,
                "three_look_at": _position3(look_at).three,
                "unity_position": _position3(follow_pos).unity,
                "unity_look_at": _position3(look_at).unity,
            },
            "carrier_fpv": {
                "type": "first_person",
                "target_player_id": int(carrier.player_id),
                "direction_source": view_direction_source,
                "fov_degrees": 88.0,
                "three_position": carrier_position.three,
                "three_look_at": _position3(look_at).three,
                "unity_position": carrier_position.unity,
                "unity_look_at": _position3(look_at).unity,
            },
        },
    }
