"""
处理真实转播画面
从标注的关键点和球员位置生成第一人称战术视角
"""

import numpy as np
import json
import cv2
import os
from src.pitch_3d import Pitch3D
from src.player_renderer import Player, Ball, TeamSide, PlayerRenderer
from src.view_transformer import ViewTransformer, CameraParams
from src.first_person_view import FirstPersonViewConverter, BroadcastFrame
from src.scene_graph import build_scene_graph
from tactical_renderer import TacticalRenderer


def load_annotations(path="annotations.json"):
    with open(path, 'r') as f:
        annotations = json.load(f)

    # 球员脚点是站位分析的基础：优先读取独立标注文件，避免继续依赖代码里的 fallback。
    player_path = "player_positions.json"
    if 'player_pixel_positions' not in annotations and os.path.exists(player_path):
        with open(player_path, 'r') as f:
            player_annotations = json.load(f)
        if player_annotations.get('image_size') == annotations.get('image_size'):
            annotations['player_pixel_positions'] = player_annotations.get('player_pixel_positions', [])

    return annotations


def estimate_camera(annotations):
    """从关键点估计相机参数（使用单应性矩阵）"""
    pts = annotations['points']
    coords = annotations['world_coords']
    img_size = tuple(annotations['image_size'])

    pts_2d = []
    pts_3d = []
    for name in pts:
        if name in coords:
            pts_2d.append(pts[name])
            pts_3d.append(coords[name])

    pts_2d = np.array(pts_2d, dtype=np.float32)
    pts_3d = np.array(pts_3d, dtype=np.float32)

    H, _ = cv2.findHomography(pts_2d, pts_3d, cv2.RANSAC, 3.0)

    cam = CameraParams(
        position=np.array([0.0, 0.0, 0.0]),
        rotation=np.array([0, 0, 0]),
        focal_length=1000,
        principal_point=(img_size[0] / 2, img_size[1] / 2),
        image_size=img_size
    )
    cam._R_override = np.eye(3)
    cam._H = H
    return cam, H


def pixel_to_world(H, px, py):
    """用单应性矩阵将像素坐标映射到球场平面 (z=0)"""
    pt = np.array([px, py, 1.0])
    wp = H @ pt
    wp /= wp[2]
    return np.array([wp[0], wp[1], 0.0])


def draw_foot_points_debug(image_path, vision_frame, output_path):
    """保存脚点调试图：检查 bbox 底点/估计脚点/世界坐标是否可信。"""
    image = cv2.imread(image_path)
    if image is None:
        return

    debug = image.copy()
    for player in vision_frame.players:
        x, y, w, h = player.bbox
        fx, fy = player.foot_pixel
        wx, wy, _ = player.world_position
        color = (0, 255, 0) if player.is_ball_carrier else (0, 220, 255)
        if player.team == "away":
            color = (0, 145, 255)
        elif player.team == "referee":
            color = (60, 60, 60)

        cv2.rectangle(debug, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.line(debug, (int(x + w * 0.5), int(y + h)), (int(round(fx)), int(round(fy))), (255, 255, 255), 1)
        cv2.drawMarker(debug, (int(round(fx)), int(round(fy))), color, cv2.MARKER_CROSS, 18, 2)
        label = f"#{player.player_id} {player.team} fp={player.foot_source} fc={player.foot_confidence:.2f} ({wx:.1f},{wy:.1f})"
        cv2.putText(debug, label, (int(x), max(18, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)
        cv2.putText(debug, label, (int(x), max(18, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    if vision_frame.ball is not None:
        bx, by = vision_frame.ball.pixel
        wx, wy, _ = vision_frame.ball.world_position
        cv2.circle(debug, (int(round(bx)), int(round(by))), 12, (255, 255, 255), 2)
        cv2.putText(debug, f"ball ({wx:.1f},{wy:.1f})", (int(bx) + 14, int(by) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(debug, f"ball ({wx:.1f},{wy:.1f})", (int(bx) + 14, int(by) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imwrite(output_path, debug)


def estimate_player_positions(image_path, H, annotations):
    """
    用单应性矩阵将球员像素坐标直接映射到球场平面
    """
    img = cv2.imread(image_path)
    if img is None:
        print("无法加载图像，使用估算位置")
        return get_estimated_positions()

    player_image_positions = annotations.get('player_pixel_positions') or [
        (858, 245), (829, 196), (600, 234), (1099, 165), (1065, 390),
        (1369, 287), (1442, 566), (678, 533), (230, 196), (80, 554),
        (532, 370), (546, 253), (639, 217), (796, 157), (808, 235),
        (881, 206), (1014, 215), (917, 353), (1340, 417),
    ]
    ball_carrier_idx = annotations.get('players', {}).get('ball_carrier_idx', 0)

    # 阵营分配 (基于原图观察)
    # 9号(230,196) = 门将(橙), 10号(80,554) = 左下角(橙)
    # 3,11,12,13,14,15 号 = 左半场偏橙
    # 1,2,4,5,6,7,8,16,17,18,19 号 = 右半场偏白
    orange_indices = [0, 1, 2, 9, 10, 11, 12]   # 门将 + 左半场
    white_indices = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]  # 右半场

    players = []

    for idx, (px, py) in enumerate(player_image_positions):
        world_pos = pixel_to_world(H, px, py)

        if idx in orange_indices:
            team = TeamSide.AWAY
        elif idx in white_indices:
            team = TeamSide.HOME
        else:
            team = TeamSide.HOME

        direction = np.array([1, 0, 0])
        if world_pos[0] > 0:
            direction = np.array([-1, 0, 0])

        player_id = idx + 1
        is_ball_carrier = (idx == ball_carrier_idx)

        players.append(Player(
            player_id=player_id,
            position=world_pos,
            direction=direction,
            team=team,
            is_ball_carrier=is_ball_carrier
        ))

    return players


def render_reprojected_ground_view(source_image, H, target_camera, output_size,
                                   pitch_margin=2.0):
    """把原图中的球场地面纹理重投影到目标相机视角。"""
    H_inv = np.linalg.inv(H)
    W, H_out = output_size
    src_h, src_w = source_image.shape[:2]

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H_out, dtype=np.float32))
    fx = float(target_camera.focal_length)
    fy = float(target_camera.focal_length)
    cx, cy = target_camera.principal_point

    ray_cam = np.stack([
        (xs - cx) / fx,
        -(ys - cy) / fy,
        np.ones_like(xs),
    ], axis=-1)
    ray_cam /= np.linalg.norm(ray_cam, axis=-1, keepdims=True)

    R_c2w = target_camera.get_rotation_matrix()
    ray_world = ray_cam @ R_c2w.T
    ray_z = ray_world[..., 2]

    valid = ray_z < -1e-6
    t = np.zeros_like(ray_z, dtype=np.float32)
    t[valid] = -float(target_camera.position[2]) / ray_z[valid]
    valid &= t > 0

    world = target_camera.position.reshape(1, 1, 3) + ray_world * t[..., None]
    wx = world[..., 0]
    wy = world[..., 1]
    valid &= (wx >= -52.5 - pitch_margin) & (wx <= 52.5 + pitch_margin)
    valid &= (wy >= -34.0 - pitch_margin) & (wy <= 34.0 + pitch_margin)

    denom = H_inv[2, 0] * wx + H_inv[2, 1] * wy + H_inv[2, 2]
    valid &= np.abs(denom) > 1e-6
    map_x = (H_inv[0, 0] * wx + H_inv[0, 1] * wy + H_inv[0, 2]) / denom
    map_y = (H_inv[1, 0] * wx + H_inv[1, 1] * wy + H_inv[1, 2]) / denom
    valid &= (map_x >= 0) & (map_x < src_w - 1) & (map_y >= 0) & (map_y < src_h - 1)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    ground = cv2.remap(source_image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 120, 50))
    ground[~valid] = (30, 120, 50)
    return ground, valid


def draw_position_anchors(image, players, camera_params):
    """在目标视角中标注球员脚点，便于检查站位是否偏移。"""
    vt = ViewTransformer()
    h, w = image.shape[:2]
    for player in players:
        foot = player.position.copy()
        foot[2] = 0.0
        cp = vt.world_to_camera([foot], camera_params)[0]
        if cp is None or cp[2] <= 0:
            continue
        ip = vt.camera_to_image([cp], camera_params, clip_to_bounds=False)[0]
        if ip is None:
            continue
        u, v = int(round(ip[0])), int(round(ip[1]))
        if u < -20 or u >= w + 20 or v < -20 or v >= h + 20:
            continue
        color = (0, 255, 0) if player.is_ball_carrier else (255, 255, 255)
        cv2.circle(image, (u, v), 5, (0, 0, 0), -1)
        cv2.circle(image, (u, v), 4, color, -1)
        cv2.putText(image, f"#{player.player_id}", (u + 6, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(image, f"#{player.player_id}", (u + 6, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def render_relative_position_map(players, ball, ball_carrier, direction, output_size=(900, 900), radius=38.0):
    """渲染以持球者为原点、视野方向为上方的相对站位图。"""
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    canvas[:] = (28, 95, 38)
    w, h = output_size
    center = np.array([w / 2, h * 0.68], dtype=float)
    scale = min(w, h) * 0.42 / radius

    fwd = np.array([direction[0], direction[1]], dtype=float)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        fwd = np.array([1.0, 0.0])
    else:
        fwd /= norm
    right = np.array([fwd[1], -fwd[0]], dtype=float)

    def rel_to_pixel(world_xy):
        rel = np.array(world_xy, dtype=float) - ball_carrier.position[:2]
        x_right = float(np.dot(rel, right))
        y_forward = float(np.dot(rel, fwd))
        px = center[0] + x_right * scale
        py = center[1] - y_forward * scale
        return int(round(px)), int(round(py)), x_right, y_forward

    # 距离环与方向轴
    for meters in (5, 10, 20, 30):
        cv2.circle(canvas, tuple(center.astype(int)), int(round(meters * scale)), (60, 130, 70), 1)
        cv2.putText(canvas, f"{meters}m", (int(center[0] + meters * scale + 4), int(center[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)
    cv2.arrowedLine(canvas, tuple(center.astype(int)), (int(center[0]), int(center[1] - 16 * scale)),
                    (255, 255, 255), 2, tipLength=0.12)
    cv2.putText(canvas, "FORWARD", (int(center[0] + 8), int(center[1] - 16 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.line(canvas, (0, int(center[1])), (w, int(center[1])), (45, 115, 55), 1)
    cv2.line(canvas, (int(center[0]), 0), (int(center[0]), h), (45, 115, 55), 1)

    relative_rows = []
    sorted_players = sorted(
        players,
        key=lambda p: 0 if p.is_ball_carrier else np.linalg.norm(p.position[:2] - ball_carrier.position[:2])
    )

    colors = {
        TeamSide.HOME: (255, 80, 40),
        TeamSide.AWAY: (40, 140, 255),
        TeamSide.REFEREE: (0, 0, 0),
    }
    label_slots = []
    for player in sorted_players:
        px, py, x_right, y_forward = rel_to_pixel(player.position[:2])
        dist = float(np.hypot(x_right, y_forward))
        relative_rows.append({
            "player_id": int(player.player_id),
            "team": player.team.value,
            "is_ball_carrier": bool(player.is_ball_carrier),
            "right_m": round(x_right, 2),
            "forward_m": round(y_forward, 2),
            "distance_m": round(dist, 2),
        })
        if px < -30 or px > w + 30 or py < -30 or py > h + 30:
            continue
        color = (0, 255, 0) if player.is_ball_carrier else colors.get(player.team, (255, 255, 255))
        radius_px = 12 if player.is_ball_carrier else 9
        cv2.circle(canvas, (px, py), radius_px, color, -1)
        cv2.circle(canvas, (px, py), radius_px, (255, 255, 255), 1)
        label_x = px + 10
        label_y = py - 8
        for slot_x, slot_y in label_slots:
            if abs(label_x - slot_x) < 52 and abs(label_y - slot_y) < 20:
                label_y += 18
        label_slots.append((label_x, label_y))
        cv2.putText(canvas, f"#{player.player_id}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(canvas, f"{dist:.1f}m", (label_x, label_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 240, 220), 1)

    if ball is not None:
        bx, by, _, _ = rel_to_pixel(ball.position[:2])
        cv2.circle(canvas, (bx, by), 5, (255, 255, 255), -1)
        cv2.circle(canvas, (bx, by), 7, (0, 0, 0), 1)

    cv2.rectangle(canvas, (12, 12), (360, 78), (10, 45, 18), -1)
    cv2.putText(canvas, f"Relative positions from carrier #{ball_carrier.player_id}", (22, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(canvas, "x=right, y=forward", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 200), 1)

    return canvas, relative_rows


def render_carrier_25d_view(players, ball, ball_carrier, direction, output_size=(1280, 720), max_forward=45.0, max_side=34.0):
    """渲染站位优先的持球者 2.5D 视角。强调可读性，不追求强透视压缩。"""
    w, h = output_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (24, 98, 36)

    fwd = np.array([direction[0], direction[1]], dtype=float)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        fwd = np.array([1.0, 0.0])
    else:
        fwd /= norm
    right = np.array([fwd[1], -fwd[0]], dtype=float)
    carrier_xy = np.array(ball_carrier.position[:2], dtype=float)

    origin = np.array([w * 0.50, h * 0.76], dtype=float)
    top_anchor = np.array([w * 0.50, h * 0.18], dtype=float)
    camera_back = 6.0
    side_scale = w * 0.40
    forward_scale = h * 0.50 / max_forward

    def world_to_local(world_xy):
        rel = np.array(world_xy, dtype=float) - carrier_xy
        return float(np.dot(rel, right)), float(np.dot(rel, fwd))

    def local_to_screen(x_right, y_forward, z=0.0):
        depth = max(0.0, y_forward)
        perspective = 1.0 / (1.0 + depth / camera_back)
        sx = origin[0] + x_right * (side_scale * perspective / camera_back)
        sy = origin[1] - depth * forward_scale * 0.92
        sy -= z * 26.0 * perspective
        return np.array([sx, sy], dtype=float)

    def draw_line_local(a, b, color=(210, 245, 210), thickness=2):
        samples = []
        for t in np.linspace(0.0, 1.0, 80):
            x = a[0] * (1 - t) + b[0] * t
            y = a[1] * (1 - t) + b[1] * t
            if y < -1.0 or y > max_forward:
                continue
            if abs(x) > max_side * 1.25:
                continue
            p = local_to_screen(x, y)
            if -200 <= p[0] <= w + 200 and -100 <= p[1] <= h + 100:
                samples.append(tuple(np.round(p).astype(int)))
        if len(samples) >= 2:
            cv2.polylines(canvas, [np.array(samples, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    def pitch_world_to_local(x, y):
        return world_to_local((x, y))

    # 地面渐变与网格
    for row in range(h):
        alpha = row / max(1, h - 1)
        canvas[row, :, :] = (int(22 + 10 * alpha), int(86 + 18 * alpha), int(31 + 6 * alpha))
    for y_forward in [5, 10, 15, 20, 30, 40]:
        draw_line_local((-max_side, y_forward), (max_side, y_forward), (50, 135, 60), 1)
    for x_right in range(-30, 31, 10):
        draw_line_local((x_right, 0), (x_right, max_forward), (45, 125, 55), 1)

    # 真实球场线：边线、中线、大禁区、半场圈等，全部先转为持球者局部坐标再投影。
    pitch_x = [-52.5, 52.5]
    pitch_y = [-34.0, 34.0]
    world_segments = [
        ((-52.5, -34.0), (52.5, -34.0)), ((-52.5, 34.0), (52.5, 34.0)),
        ((-52.5, -34.0), (-52.5, 34.0)), ((52.5, -34.0), (52.5, 34.0)),
        ((0.0, -34.0), (0.0, 34.0)),
        ((-52.5, -20.16), (-36.0, -20.16)), ((-36.0, -20.16), (-36.0, 20.16)), ((-36.0, 20.16), (-52.5, 20.16)),
        ((52.5, -20.16), (36.0, -20.16)), ((36.0, -20.16), (36.0, 20.16)), ((36.0, 20.16), (52.5, 20.16)),
    ]
    for a, b in world_segments:
        draw_line_local(pitch_world_to_local(*a), pitch_world_to_local(*b), (230, 250, 230), 2)
    center_circle = []
    for ang in np.linspace(0, 2 * np.pi, 180):
        wx = 9.15 * np.cos(ang)
        wy = 9.15 * np.sin(ang)
        lx, ly = pitch_world_to_local(wx, wy)
        if 0.0 <= ly <= max_forward and abs(lx) <= max_side * 1.25:
            center_circle.append(tuple(np.round(local_to_screen(lx, ly)).astype(int)))
    if len(center_circle) >= 2:
        cv2.polylines(canvas, [np.array(center_circle, dtype=np.int32)], False, (230, 250, 230), 2, cv2.LINE_AA)

    colors = {
        TeamSide.HOME: (255, 75, 35),
        TeamSide.AWAY: (35, 135, 255),
        TeamSide.REFEREE: (15, 15, 15),
    }

    drawable = []
    for player in players:
        x_right, y_forward = world_to_local(player.position[:2])
        if y_forward < 0.0 or y_forward > max_forward or abs(x_right) > max_side:
            continue
        drawable.append((y_forward, x_right, player))
    drawable.sort(key=lambda item: (item[0], abs(item[1])), reverse=True)  # 远处先画，近处后画

    for y_forward, x_right, player in drawable:
        feet = local_to_screen(x_right, y_forward, 0.0)
        depth = max(0.5, y_forward + camera_back)
        body_h = int(np.clip(170 / depth + 20, 20, 72))
        body_w = int(max(10, body_h * 0.40))
        base = tuple(np.round(feet).astype(int))
        shadow_w = int(body_w * 1.3)
        cv2.ellipse(canvas, base, (shadow_w, max(3, body_w // 4)), 0, 0, 360, (10, 55, 20), -1, cv2.LINE_AA)
        color = (0, 255, 0) if player.is_ball_carrier else colors.get(player.team, (240, 240, 240))
        x0, y0 = int(feet[0] - body_w / 2), int(feet[1] - body_h)
        x1, y1 = int(feet[0] + body_w / 2), int(feet[1])
        cv2.rectangle(canvas, (x0 + 3, y0 + 3), (x1 + 3, y1 + 3), (0, 45, 20), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
        head_r = max(4, body_w // 3)
        cv2.circle(canvas, (int(feet[0]), y0 - head_r), head_r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(feet[0]), y0 - head_r), head_r, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"#{player.player_id}"
        cv2.putText(canvas, label, (x0 - 2, y0 - head_r * 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(canvas, label, (x0 - 2, y0 - head_r * 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if ball is not None:
        bx, by = world_to_local(ball.position[:2])
        if -3.0 <= by <= max_forward and abs(bx) <= max_side:
            bp = local_to_screen(bx, by, 0.15)
            depth = max(0.5, by + camera_back)
            r = int(np.clip(36 / depth + 2, 4, 10))
            cv2.circle(canvas, tuple(np.round(bp).astype(int)), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(np.round(bp).astype(int)), r, (245, 245, 245), -1, cv2.LINE_AA)

    cv2.line(canvas, tuple(np.round(top_anchor).astype(int)), (w - 40, int(h * 0.22)), (70, 155, 80), 1)
    cv2.line(canvas, tuple(np.round(top_anchor).astype(int)), (40, int(h * 0.22)), (70, 155, 80), 1)

    # HUD 和图例
    cv2.rectangle(canvas, (16, 16), (440, 82), (8, 35, 14), -1)
    cv2.putText(canvas, f"2.5D Carrier View  |  carrier #{ball_carrier.player_id}", (28, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(canvas, "geometry-rendered from pitch coordinates, not image warp", (28, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (195, 230, 195), 1)
    cv2.circle(canvas, (w - 220, 36), 7, (0, 255, 0), -1)
    cv2.putText(canvas, "carrier", (w - 206, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 255, 240), 1)
    cv2.circle(canvas, (w - 135, 36), 7, (35, 135, 255), -1)
    cv2.putText(canvas, "away", (w - 121, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 255, 240), 1)
    cv2.circle(canvas, (w - 74, 36), 7, (255, 75, 35), -1)
    cv2.putText(canvas, "home", (w - 60, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 255, 240), 1)

    return canvas


def render_broadcast_relative_map(players, ball_carrier, output_size=(900, 650)):
    """渲染保持原始电视画面左右/上下关系的相对站位图。"""
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    canvas[:] = (28, 95, 38)
    w, h = output_size

    if not hasattr(ball_carrier, "foot_pixel"):
        return canvas, []

    carrier_px = np.array(ball_carrier.foot_pixel, dtype=float)
    points = []
    for player in players:
        if not hasattr(player, "foot_pixel"):
            continue
        delta_px = np.array(player.foot_pixel, dtype=float) - carrier_px
        delta_world = player.position[:2] - ball_carrier.position[:2]
        dist_m = float(np.linalg.norm(delta_world))
        points.append((player, delta_px, dist_m))

    if not points:
        return canvas, []

    max_abs_x = max(abs(float(delta[0])) for _, delta, _ in points) or 1.0
    max_abs_y = max(abs(float(delta[1])) for _, delta, _ in points) or 1.0
    scale = min(w * 0.42 / max_abs_x, h * 0.38 / max_abs_y, 1.2)
    center = np.array([w / 2, h * 0.58], dtype=float)

    cv2.rectangle(canvas, (12, 12), (480, 82), (10, 45, 18), -1)
    cv2.putText(canvas, f"Broadcast-relative positions from carrier #{ball_carrier.player_id}", (22, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(canvas, "keeps original image left/right/up/down", (22, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 200), 1)
    cv2.line(canvas, (0, int(center[1])), (w, int(center[1])), (45, 115, 55), 1)
    cv2.line(canvas, (int(center[0]), 0), (int(center[0]), h), (45, 115, 55), 1)
    cv2.putText(canvas, "image up", (int(center[0] + 8), 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 240, 220), 1)
    cv2.putText(canvas, "image right", (w - 120, int(center[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 240, 220), 1)

    colors = {
        TeamSide.HOME: (255, 80, 40),
        TeamSide.AWAY: (40, 140, 255),
        TeamSide.REFEREE: (0, 0, 0),
    }
    rows = []
    label_slots = []
    for player, delta_px, dist_m in sorted(points, key=lambda item: item[2]):
        px = int(round(center[0] + delta_px[0] * scale))
        py = int(round(center[1] + delta_px[1] * scale))
        rows.append({
            "player_id": int(player.player_id),
            "team": player.team.value,
            "is_ball_carrier": bool(player.is_ball_carrier),
            "image_dx_px": round(float(delta_px[0]), 1),
            "image_dy_px": round(float(delta_px[1]), 1),
            "distance_m": round(dist_m, 2),
        })
        if px < -30 or px > w + 30 or py < -30 or py > h + 30:
            continue
        color = (0, 255, 0) if player.is_ball_carrier else colors.get(player.team, (255, 255, 255))
        radius_px = 12 if player.is_ball_carrier else 9
        cv2.circle(canvas, (px, py), radius_px, color, -1)
        cv2.circle(canvas, (px, py), radius_px, (255, 255, 255), 1)
        label_x, label_y = px + 10, py - 8
        for slot_x, slot_y in label_slots:
            if abs(label_x - slot_x) < 52 and abs(label_y - slot_y) < 20:
                label_y += 18
        label_slots.append((label_x, label_y))
        cv2.putText(canvas, f"#{player.player_id}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(canvas, f"{dist_m:.1f}m", (label_x, label_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 240, 220), 1)

    return canvas, rows


def filter_players_in_carrier_fov(players, ball_carrier, direction, fov_degrees=120.0):
    """只保留持球者前进方向扇区内的球员，并始终保留持球者本人。"""
    dir2 = np.array(direction[:2], dtype=float)
    norm = np.linalg.norm(dir2)
    if norm < 1e-6:
        return [ball_carrier]
    dir2 /= norm

    min_dot = np.cos(np.radians(fov_degrees / 2.0))
    selected = []
    for player in players:
        if player.player_id == ball_carrier.player_id:
            selected.append(player)
            continue
        rel = player.position[:2] - ball_carrier.position[:2]
        rel_norm = np.linalg.norm(rel)
        if rel_norm < 1e-6:
            selected.append(player)
            continue
        if float(np.dot(rel / rel_norm, dir2)) >= min_dot:
            selected.append(player)
    return selected


def get_estimated_positions():
    """如果无法加载图像，使用估算位置"""
    players = [
        Player(player_id=1, position=np.array([-48, 0, 0]), direction=np.array([1,0,0]), team=TeamSide.HOME),  # 门将
        Player(player_id=2, position=np.array([-35, -15, 0]), direction=np.array([1,0,0]), team=TeamSide.AWAY),
        Player(player_id=3, position=np.array([-30, 10, 0]), direction=np.array([1,0,0]), team=TeamSide.AWAY),
        Player(player_id=4, position=np.array([-20, -5, 0]), direction=np.array([1,0,0]), team=TeamSide.AWAY, is_ball_carrier=True),
        Player(player_id=5, position=np.array([-15, 15, 0]), direction=np.array([1,0,0]), team=TeamSide.AWAY),
        Player(player_id=6, position=np.array([-10, -10, 0]), direction=np.array([1,0,0]), team=TeamSide.HOME),
        Player(player_id=7, position=np.array([-5, 5, 0]), direction=np.array([1,0,0]), team=TeamSide.HOME),
        Player(player_id=8, position=np.array([0, -15, 0]), direction=np.array([1,0,0]), team=TeamSide.AWAY),
        Player(player_id=9, position=np.array([5, 10, 0]), direction=np.array([1,0,0]), team=TeamSide.HOME),
        Player(player_id=10, position=np.array([10, -5, 0]), direction=np.array([1,0,0]), team=TeamSide.HOME),
    ]
    return players


def build_follow_camera(carrier_pos, direction, img_size,
                        camera_height=14.0, behind_dist=18.0,
                        look_ahead=16.0, fov=92.0):
    """按给定方向构造跟拍相机。"""
    dir3 = np.array([direction[0], direction[1], 0.0], dtype=float)
    n = np.linalg.norm(dir3[:2])
    if n < 1e-6:
        dir3 = np.array([0.0, -1.0, 0.0], dtype=float)
    else:
        dir3 /= n

    carrier_xy = carrier_pos[:2]
    cam_xy = carrier_xy - dir3[:2] * behind_dist
    cam_pos = np.array([cam_xy[0], cam_xy[1], camera_height], dtype=float)

    target_xy = carrier_xy + dir3[:2] * look_ahead
    target = np.array([target_xy[0], target_xy[1], 0.0], dtype=float)

    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    R_w2c = np.vstack([right, up, fwd])
    focal = img_size[0] / (2 * np.tan(np.radians(fov / 2)))
    cam = CameraParams(
        position=cam_pos,
        rotation=np.zeros(3),
        focal_length=focal,
        principal_point=(img_size[0] / 2, img_size[1] * 0.58),
        image_size=img_size
    )
    cam._R_override = R_w2c
    return cam, dir3


def pick_best_camera_direction(players, ball_carrier, img_size, centroid):
    """扫描方向并选择可见球员最多、分布更均衡的相机方向。"""
    vt = ViewTransformer()
    candidates = np.arange(0, 360, 12)
    best = None
    best_score = -1e9

    center_vec = centroid[:2] - ball_carrier.position[:2]
    center_norm = np.linalg.norm(center_vec)
    if center_norm > 1e-6:
        center_dir = center_vec / center_norm
    else:
        center_dir = np.array([0.0, -1.0])

    for deg in candidates:
        rad = np.radians(deg)
        d = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=float)
        cam, dir3 = build_follow_camera(ball_carrier.position, d, img_size)

        cam_pts = vt.world_to_camera([p.position for p in players], cam)
        img_pts = vt.camera_to_image(cam_pts, cam, clip_to_bounds=False)

        in_count = 0
        front_count = 0
        center_penalty = 0.0
        for p, cp, ip in zip(players, cam_pts, img_pts):
            if cp is None or cp[2] <= 0 or ip is None:
                continue
            u, v = ip
            in_frame = (0 <= u < img_size[0] and 0 <= v < img_size[1])
            if in_frame:
                in_count += 1
                vec = p.position - ball_carrier.position
                if np.dot(vec[:2], dir3[:2]) > 0:
                    front_count += 1
                center_penalty += abs(u - img_size[0] * 0.5) / img_size[0]

        if in_count > 0:
            center_penalty /= in_count

        # 持球者屏幕位置约束：希望在中下区域，避免跑到顶部
        c_cp = vt.world_to_camera([ball_carrier.position], cam)[0]
        carrier_term = 0.0
        if c_cp is not None and c_cp[2] > 0:
            c_ip = vt.camera_to_image([c_cp], cam, clip_to_bounds=False)[0]
            if c_ip is not None:
                cu, cv = c_ip
                if 0 <= cu < img_size[0] and 0 <= cv < img_size[1]:
                    du = abs(cu - img_size[0] * 0.5) / img_size[0]
                    dv = abs(cv - img_size[1] * 0.70) / img_size[1]
                    carrier_term = 4.0 - (du * 8.0 + dv * 10.0)

        align = float(np.dot(dir3[:2], center_dir))
        center_prior = align * 2.0
        score = in_count * 6.0 + front_count * 1.0 - center_penalty * 2.0 + carrier_term + center_prior

        if score > best_score:
            best_score = score
            best = (cam, dir3, deg, in_count, front_count)

    return best


def pick_ground_reprojection_camera(source_image, H, players, ball_carrier, img_size, play_direction):
    """选择路线B相机：朝向跟随持球者，只优化高度/距离以保证地面覆盖。"""
    vt = ViewTransformer()
    direction = np.array(play_direction, dtype=float)
    direction_norm = np.linalg.norm(direction[:2])
    if direction_norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        direction = direction / direction_norm
    deg = np.degrees(np.arctan2(direction[1], direction[0]))

    best = None
    best_score = -1e9
    small_size = (img_size[0] // 4, img_size[1] // 4)

    for height in (10.0, 14.0, 20.0, 28.0):
        for behind_dist in (0.0, 6.0, 12.0, 18.0):
            for look_ahead in (12.0, 20.0, 32.0):
                score_camera, _ = build_follow_camera(
                    ball_carrier.position,
                    direction,
                    small_size,
                    camera_height=height,
                    behind_dist=behind_dist,
                    look_ahead=look_ahead,
                    fov=92.0
                )
                _, valid_mask = render_reprojected_ground_view(source_image, H, score_camera, small_size)
                coverage = float(valid_mask.mean())

                full_camera, dir3 = build_follow_camera(
                    ball_carrier.position,
                    direction,
                    img_size,
                    camera_height=height,
                    behind_dist=behind_dist,
                    look_ahead=look_ahead,
                    fov=92.0
                )
                cam_points = vt.world_to_camera([p.position for p in players], full_camera)
                img_points = vt.camera_to_image(cam_points, full_camera, clip_to_bounds=False)
                in_frame = 0
                for cp, ip in zip(cam_points, img_points):
                    if cp is None or cp[2] <= 0 or ip is None:
                        continue
                    u, v = ip
                    if 0 <= u < img_size[0] and 0 <= v < img_size[1]:
                        in_frame += 1

                carrier_cp = vt.world_to_camera([ball_carrier.position], full_camera)[0]
                carrier_score = -10.0
                if carrier_cp is not None and carrier_cp[2] > 0:
                    carrier_ip = vt.camera_to_image([carrier_cp], full_camera, clip_to_bounds=False)[0]
                    if carrier_ip is not None:
                        cu, cv = carrier_ip
                        du = abs(cu - img_size[0] * 0.5) / img_size[0]
                        dv = abs(cv - img_size[1] * 0.78) / img_size[1]
                        carrier_score = 2.0 - du * 4.0 - dv * 6.0

                score = coverage * 100.0 + in_frame * 2.0 + carrier_score
                if score > best_score:
                    best_score = score
                    best = (full_camera, dir3, deg, coverage, in_frame, height, behind_dist, look_ahead)

    return best


def draw_carrier_hud(image, ball_carrier, camera_params):
    """在第一人称图上给出持球者可见性提示（画内标记+画外箭头+HUD文字）。"""
    h, w = image.shape[:2]
    vt = ViewTransformer()

    label = f"BALL CARRIER #{ball_carrier.player_id}"
    cv2.rectangle(image, (16, 16), (330, 52), (20, 20, 20), -1)
    cv2.putText(image, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cp = vt.world_to_camera([ball_carrier.position], camera_params)[0]
    if cp is None or cp[2] <= 0:
        cv2.putText(image, "OUT OF VIEW (BEHIND)", (24, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        return image

    ip = vt.camera_to_image([cp], camera_params, clip_to_bounds=False)[0]
    if ip is None:
        return image

    u, v = float(ip[0]), float(ip[1])
    if 0 <= u < w and 0 <= v < h:
        ui, vi = int(u), int(v)
        cv2.circle(image, (ui, vi), 22, (0, 255, 0), 2)
        cv2.putText(image, "CARRIER", (ui + 12, vi - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return image

    cx, cy = w * 0.5, h * 0.5
    dx, dy = u - cx, v - cy
    n = np.hypot(dx, dy)
    if n < 1e-6:
        return image
    dx, dy = dx / n, dy / n

    margin = 28
    tx = int(np.clip(cx + dx * (w * 0.46), margin, w - margin))
    ty = int(np.clip(cy + dy * (h * 0.46), margin, h - margin))
    px = int(tx - dx * 22)
    py = int(ty - dy * 22)
    cv2.arrowedLine(image, (px, py), (tx, ty), (0, 255, 0), 3, tipLength=0.45)
    cv2.putText(image, "CARRIER", (tx - 45, ty - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def render_broadcast_view(img_size, players, ball, H, pitch):
    """用 homography 渲染转播视角（像素→世界→像素）"""
    H_inv = np.linalg.inv(H)
    canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    canvas[:] = (30, 120, 50)

    def project_world_to_pixel(wx, wy):
        pt = H_inv @ np.array([wx, wy, 1.0])
        pt /= pt[2]
        return int(pt[0]), int(pt[1])

    # 球场线条
    kp = pitch.get_pitch_key_points()
    lines = [
        ([0, 1, 3, 2, 0], False),
        ([45, 46], False),
        ([4, 5, 7, 6, 4], False),
        ([8, 9, 11, 10, 8], False),
    ]
    for indices, closed in lines:
        pts = []
        for i in indices:
            wx, wy = kp[i][0], kp[i][1]
            pts.append(project_world_to_pixel(wx, wy))
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], closed, (255, 255, 255), 2)

    center_circle = kp[12:44]
    cc_pts = [project_world_to_pixel(p[0], p[1]) for p in center_circle]
    cv2.polylines(canvas, [np.array(cc_pts, dtype=np.int32)], True, (255, 255, 255), 1)

    # 球员
    colors = {
        'home': (0, 100, 255),
        'away': (255, 50, 50),
        'referee': (255, 255, 0),
    }
    for player in players:
        u, v = project_world_to_pixel(player.position[0], player.position[1])
        color = (0, 255, 0) if player.is_ball_carrier else colors.get(player.team.value, (255, 255, 255))
        cv2.circle(canvas, (u, v), 10, color, -1)
        cv2.circle(canvas, (u, v), 10, (255, 255, 255), 1)
        cv2.putText(canvas, f"#{player.player_id}", (u - 15, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 球
    bu, bv = project_world_to_pixel(ball.position[0], ball.position[1])
    cv2.circle(canvas, (bu, bv), 5, (255, 255, 255), -1)

    return canvas


def process_real_image(image_path, annotation_path="annotations.json", output_dir="output_real", auto_detect=False,
                       object_model_path="yolov8n.pt", with_ground_reprojection=False,
                       use_color_fallback=True):
    print("=" * 60)
    print("真实转播画面处理")
    print("=" * 60)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    if auto_detect:
        from src.vision_pipeline import FootballVisionPipeline, save_vision_frame

        img_size = (img.shape[1], img.shape[0])
        print(f"\n图像: {image_path} ({img_size[0]}x{img_size[1]})")
        print("\n自动识别球场、球员和足球...")
        pipeline = FootballVisionPipeline(
            object_model_path=object_model_path,
            use_color_fallback=use_color_fallback,
        )
        vision_frame = pipeline.process(image_path)
        camera, players, ball = pipeline.to_render_objects(vision_frame)
        H = vision_frame.pitch.homography

        os.makedirs(output_dir, exist_ok=True)
        save_vision_frame(vision_frame, os.path.join(output_dir, "vision_frame.json"))
        pipeline.draw_debug_overlay(image_path, vision_frame, os.path.join(output_dir, "vision_overlay.png"))

        print(f"  球场标定: {vision_frame.pitch.method}")
        print(f"  识别球员: {len(players)}")
        print(f"  识别足球: {'yes' if vision_frame.ball else 'no'}")
        print(f"  持球者: #{vision_frame.ball_carrier_id}" if vision_frame.ball_carrier_id else "  持球者: 未确定")
    else:
        # 加载标注
        annotations = load_annotations(annotation_path)
        img_size = tuple(annotations['image_size'])
        print(f"\n图像: {image_path} ({img_size[0]}x{img_size[1]})")

        # 估计相机
        print("\n估计相机参数...")
        camera, H = estimate_camera(annotations)

        # 验证标定
        print("\n验证标定误差:")
        pts = annotations['points']
        coords = annotations['world_coords']
        total_error = 0

        for name in pts.keys():
            if name in coords:
                ip = pts[name]
                wp = coords[name]
                proj_wp = pixel_to_world(H, ip[0], ip[1])
                error = np.sqrt((proj_wp[0] - wp[0])**2 + (proj_wp[1] - wp[1])**2)
                total_error += error
                print(f"  {name}: 世界({wp[0]:.1f},{wp[1]:.1f}) 映射({proj_wp[0]:.1f},{proj_wp[1]:.1f}) 误差={error:.2f}m")

        print(f"  平均误差: {total_error/len(pts):.3f}m")

        # 估计球员位置
        print("\n估计球员位置...")
        players = estimate_player_positions(image_path, H, annotations)
        print(f"  检测到 {len(players)} 名球员")

        # 找到持球队员
        ball_carrier = next((p for p in players if p.is_ball_carrier), None)
        if ball_carrier:
            ball = Ball(position=ball_carrier.position + np.array([0.5, 0.2, 0.05]))
            print(f"  持球队员: #{ball_carrier.player_id}")
        else:
            players[0].is_ball_carrier = True
            ball_carrier = players[0]
            ball = Ball(position=ball_carrier.position + np.array([0.5, 0.2, 0.05]))
            print(f"  默认持球队员: #{ball_carrier.player_id}")
    
    # 创建球场
    pitch = Pitch3D()
    
    # 执行转换
    print("\n执行视角转换...")

    # 过滤掉坐标明显出界的球员（单应性边缘误差）
    valid_players = [p for p in players if abs(p.position[0]) < 55 and abs(p.position[1]) < 38]
    invalid_count = len(players) - len(valid_players)
    if invalid_count:
        print(f"  过滤掉 {invalid_count} 个坐标出界的球员")
    players = valid_players
    if not players:
        raise ValueError("没有可用球员检测结果；请换用更强权重、降低检测阈值或启用颜色 fallback")

    ball_carrier = next((p for p in players if p.is_ball_carrier), players[0])

    # 球员重心（仅用于调试）
    positions = np.array([p.position for p in players])
    centroid = positions.mean(axis=0).copy()
    centroid[2] = 0.0
    print(f"  球员重心: ({centroid[0]:.1f}, {centroid[1]:.1f})")

    # 相机仍沿用之前“整体视角/球场线更正确”的自动选择；站位筛选独立使用该方向的120度前方范围。
    is_penalty_kick = ball is not None and abs(float(ball.position[0]) - 41.5) < 0.2 and abs(float(ball.position[1])) < 0.2
    if is_penalty_kick:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        fp_camera, direction = build_follow_camera(ball_carrier.position, direction, img_size)
        best_deg, in_count, front_count = 0.0, len(players), sum(1 for p in players if p.position[0] > ball_carrier.position[0])
    else:
        cam_pick = pick_best_camera_direction(players, ball_carrier, img_size, centroid)
        fp_camera, direction, best_deg, in_count, front_count = cam_pick
    ball_carrier.direction = direction
    focus_players = filter_players_in_carrier_fov(players, ball_carrier, direction, fov_degrees=120.0)
    print(f"  最佳朝向: {best_deg:.0f}°")
    print(f"  朝向评分: 画内球员 {in_count}，前方球员 {front_count}")
    print(f"  视野方向: ({direction[0]:.2f}, {direction[1]:.2f})")
    print(f"  120°前进视野球员: {len(focus_players)}/{len(players)} -> {[p.player_id for p in focus_players]}")
    print(f"  相机位置: ({fp_camera.position[0]:.1f}, {fp_camera.position[1]:.1f}, {fp_camera.position[2]:.1f})")

    # 调试：俯仰角
    look_vec = fp_camera._R_override[2]
    pitch_angle = np.degrees(np.arctan2(look_vec[2], np.linalg.norm(look_vec[:2])))
    print(f"  俯仰角: {pitch_angle:.1f}° (负=向下，PES 约 -20~-35°)")

    FOV = 92.0
    CAM_HEIGHT = float(fp_camera.position[2])

    converter = FirstPersonViewConverter(
        pitch=pitch,
        output_size=img_size,
        first_person_fov=FOV,
        camera_height=CAM_HEIGHT
    )

    frame = BroadcastFrame(
        source_camera=camera,
        players=players,
        ball=ball
    )

    first_person_image = converter.renderer.render_scene(
        camera_params=fp_camera,
        players=focus_players,
        ball=ball,
        pitch_key_points=pitch.get_pitch_key_points(),
        show_view_cone=False,
        fov_angle=FOV
    )
    wireframe_image = converter.renderer.generate_wireframe_scene(
        camera_params=fp_camera,
        players=focus_players,
        ball=ball,
        pitch_key_points=pitch.get_pitch_key_points()
    )

    # 持球者可见性 HUD
    first_person_image = draw_carrier_hud(first_person_image, ball_carrier, fp_camera)

    # 计算可见球员（简单用 world_to_camera 判断 z>0）
    from src.view_transformer import ViewTransformer as _VT
    _vt = _VT()
    visible_players = []
    for p in players:
        cam_pts = _vt.world_to_camera([p.position], fp_camera)
        if cam_pts[0] is not None and cam_pts[0][2] > 0:
            visible_players.append(p)
    print(f"  可见球员数: {len(visible_players)}")
    
    # 保存结果
    print("\n保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    for stale_name in (
        "first_person_view.png", "ground_reprojected_only.png", "ground_reprojected_view.png",
        "wireframe_view.png", "tactical_first_person.png", "overlay.png", "broadcast_view.png",
        "top_down_view.png", "grid_overlay.png", "foot_points_debug.png",
        "keypoints_detected.png", "relative_positions.png", "relative_positions.json",
        "broadcast_relative_positions.png", "broadcast_relative_positions.json", "carrier_25d_view.png",
        "scene_graph.json",
    ):
        stale_path = os.path.join(output_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    if auto_detect and 'vision_frame' in locals():
        draw_foot_points_debug(image_path, vision_frame, os.path.join(output_dir, "foot_points_debug.png"))
    
    # 1. 第一人称视角（从持球队员视角渲染）
    print("  生成第一人称视角...")
    cv2.imwrite(os.path.join(output_dir, "first_person_view.png"), first_person_image)

    if with_ground_reprojection:
        # 路线B Demo：真实地面纹理重投影 + 同一套世界坐标球员复投影。
        # 这会放大 homography/球员脚点误差，默认关闭；需要排查真实纹理投影时再打开。
        print("  生成真实地面重投影视角...")
        reprojected_ground, reprojected_mask = render_reprojected_ground_view(img, H, fp_camera, img_size)
        print(f"  真实地面覆盖: {float(reprojected_mask.mean())*100:.1f}%")
        cv2.imwrite(os.path.join(output_dir, "ground_reprojected_only.png"), reprojected_ground)
        ground_reprojected_view = converter.renderer.render_scene(
            camera_params=fp_camera,
            players=focus_players,
            ball=ball,
            pitch_key_points=pitch.get_pitch_key_points(),
            show_view_cone=False,
            fov_angle=FOV,
            background_image=reprojected_ground,
            show_ground_grid=False,
            show_pitch_lines=False
        )
        ground_reprojected_view = draw_position_anchors(ground_reprojected_view, focus_players, fp_camera)
        cv2.imwrite(os.path.join(output_dir, "ground_reprojected_view.png"), ground_reprojected_view)
    
    # 2. 线框技术视图
    if wireframe_image is not None:
        cv2.imwrite(os.path.join(output_dir, "wireframe_view.png"), wireframe_image)

    # 2b. 持球者中心相对站位图：用于检查队友/对手与持球者的相对位置
    print("  生成相对站位图...")
    relative_map, relative_rows = render_relative_position_map(players, ball, ball_carrier, direction)
    cv2.imwrite(os.path.join(output_dir, "relative_positions.png"), relative_map)
    with open(os.path.join(output_dir, "relative_positions.json"), "w", encoding="utf-8") as f:
        json.dump({
            "ball_carrier_id": int(ball_carrier.player_id),
            "direction": [float(direction[0]), float(direction[1])],
            "coordinate_system": "right_m is carrier-right, forward_m is carrier-forward",
            "players": relative_rows,
        }, f, ensure_ascii=False, indent=2)

    print("  导出通用 scene graph...")
    scene_graph = build_scene_graph(
        image_path=image_path,
        image_size=img_size,
        players=players,
        ball=ball,
        carrier=ball_carrier,
        play_direction=direction,
        pitch_method=getattr(vision_frame.pitch, "method", "manual_annotations") if auto_detect else "manual_annotations",
        homography=H,
    )
    with open(os.path.join(output_dir, "scene_graph.json"), "w", encoding="utf-8") as f:
        json.dump(scene_graph, f, ensure_ascii=False, indent=2)

    print("  生成持球者2.5D视角...")
    carrier_25d = render_carrier_25d_view(players, ball, ball_carrier, direction)
    cv2.imwrite(os.path.join(output_dir, "carrier_25d_view.png"), carrier_25d)

    broadcast_relative_map, broadcast_relative_rows = render_broadcast_relative_map(players, ball_carrier)
    cv2.imwrite(os.path.join(output_dir, "broadcast_relative_positions.png"), broadcast_relative_map)
    with open(os.path.join(output_dir, "broadcast_relative_positions.json"), "w", encoding="utf-8") as f:
        json.dump({
            "ball_carrier_id": int(ball_carrier.player_id),
            "coordinate_system": "image_dx_px/image_dy_px preserve original broadcast image directions",
            "players": broadcast_relative_rows,
        }, f, ensure_ascii=False, indent=2)
    
    # 3. 战术风格第一人称视角（在真实画面上叠加战术元素）
    print("  生成战术风格视图...")
    tactical = TacticalRenderer(img_size)
    tactical_view = tactical.render_tactical_view(
        background_image=img,
        camera_params=camera,
        players=players,
        ball_carrier=ball_carrier,
        show_view_cone=True,
        show_pass_lines=True,
        show_pitch_lines=False
    )
    cv2.imwrite(os.path.join(output_dir, "tactical_first_person.png"), tactical_view)
    
    # 4. 原始图像叠加球员位置
    H_inv = np.linalg.inv(H)
    overlay = img.copy()
    for player in players:
        wp = player.position
        pt = H_inv @ np.array([wp[0], wp[1], 1.0])
        pt /= pt[2]
        u, v = int(pt[0]), int(pt[1])
        color = (0, 255, 0) if player.is_ball_carrier else (255, 255, 0)
        cv2.circle(overlay, (u, v), 10, color, -1)
        cv2.putText(overlay, f"#{player.player_id}", (u-15, v-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(output_dir, "overlay.png"), overlay)
    
    # 5. 转播视角渲染（用 homography 投影代替 3D 相机）
    print("  生成转播视角...")
    broadcast_view = render_broadcast_view(img_size, players, ball, H, pitch)
    cv2.imwrite(os.path.join(output_dir, "broadcast_view.png"), broadcast_view)
    
    # 6. 俯视图
    top_down = CameraParams(
        position=np.array([0, 0, 60]),
        rotation=np.array([0, 0, 0]),
        focal_length=500,
        principal_point=(img_size[0]/2, img_size[1]/2),
        image_size=img_size
    )
    R_td = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    top_down._R_override = R_td
    
    top_down_img = converter.renderer.render_scene(
        camera_params=top_down,
        players=players,
        ball=ball,
        pitch_key_points=pitch.get_pitch_key_points(),
        show_view_cone=False
    )
    cv2.imwrite(os.path.join(output_dir, "top_down_view.png"), top_down_img)
    
    print(f"\n输出目录: {output_dir}/")
    print("  - first_person_view.png: 第一人称视角")
    if with_ground_reprojection:
        print("  - ground_reprojected_only.png: 真实地面重投影")
        print("  - ground_reprojected_view.png: 真实地面重投影 + 球员站位")
    print("  - wireframe_view.png: 线框技术视图")
    print("  - relative_positions.png: 持球者中心相对站位图")
    print("  - relative_positions.json: 相对站位数据")
    print("  - broadcast_relative_positions.png: 保持原图方向的相对站位图")
    print("  - tactical_first_person.png: 战术风格第一人称视角")
    print("  - overlay.png: 原始图像叠加球员位置")
    print("  - broadcast_view.png: 转播视角渲染")
    print("  - top_down_view.png: 俯视图")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="处理真实足球转播截图并生成持球者视角")
    parser.add_argument("image", nargs="?", default="input/ScreenShot_2026-06-09_001335_020.png")
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--output-dir", default="output_real")
    parser.add_argument("--auto-detect", action="store_true", help="使用自动视觉管线识别球场、球员和球")
    parser.add_argument("--object-model", default="yolov8n.pt", help="可选：本地 YOLO 人/球检测权重，如 yolov8n.pt")
    parser.add_argument("--with-ground-reprojection", action="store_true", help="额外生成真实地面纹理重投影视图")
    parser.add_argument("--no-color-fallback", action="store_true", help="只使用目标检测模型，不混入颜色启发式球员检测")
    args = parser.parse_args()

    process_real_image(
        args.image,
        args.annotations,
        args.output_dir,
        auto_detect=args.auto_detect,
        object_model_path=args.object_model,
        with_ground_reprojection=args.with_ground_reprojection,
        use_color_fallback=not args.no_color_fallback,
    )
