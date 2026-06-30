"""
球员示意图渲染器
类似SAOT(半自动越位系统)的简化球员3D表示
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class TeamSide(Enum):
    """球队阵营"""
    HOME = "home"      # 主队
    AWAY = "away"      # 客队
    REFEREE = "referee"  # 裁判


@dataclass
class Player:
    """
    球员数据
    
    简化表示：只需要位置、方向和阵营
    """
    player_id: int  # 球员编号
    position: np.ndarray  # 3D位置 [x, y, z]，z为0表示地面
    direction: np.ndarray  # 面向方向 (单位向量)
    team: TeamSide  # 阵营
    is_ball_carrier: bool = False  # 是否持球
    height: float = 1.80  # 身高 (米)
    
    def __post_init__(self):
        # 确保position是numpy数组
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        if not isinstance(self.direction, np.ndarray):
            self.direction = np.array(self.direction)


@dataclass
class Ball:
    """足球"""
    position: np.ndarray  # 3D位置
    radius: float = 0.11  # 标准足球半径


class PlayerRenderer:
    """
    球员示意图渲染器
    
    生成类似SAOT风格的简化球员和球场示意图
    """
    
    # 颜色配置 (类似SAOT风格)
    COLORS = {
        TeamSide.HOME: (0, 100, 255),      # 蓝色
        TeamSide.AWAY: (255, 50, 50),      # 红色
        TeamSide.REFEREE: (255, 255, 0),   # 黄色
        'ball': (255, 255, 255),           # 白色
        'ball_carrier_ring': (0, 255, 0),  # 持球者光环 (绿色)
        'pitch_lines': (255, 255, 255),    # 球场线条 (白色)
        'pitch_fill': (30, 120, 50),       # 草皮颜色
        'view_cone': (255, 255, 0, 50),    # 视野锥 (半透明黄色)
    }
    
    def __init__(self, image_size: Tuple[int, int] = (1920, 1080)):
        """
        Args:
            image_size: 渲染输出尺寸 (宽, 高)
        """
        self.image_size = image_size
    
    def render_scene(self, camera_params, players: List[Player], ball: Ball,
                     pitch_key_points: List[np.ndarray],
                     show_view_cone: bool = True,
                     fov_angle: float = 90.0,
                     background_image: np.ndarray = None,
                     show_ground_grid: bool = True,
                     show_pitch_lines: bool = True) -> np.ndarray:
        """
        渲染完整场景 (需要cv2)
        
        Args:
            camera_params: 相机参数
            players: 球员列表
            ball: 足球
            pitch_key_points: 球场关键点
            show_view_cone: 是否显示持球者视野锥
            fov_angle: 视野角度
            
        Returns:
            渲染的图像 (RGB numpy array)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("需要安装 opencv-python: pip install opencv-python")
        
        from src.view_transformer import ViewTransformer
        
        transformer = ViewTransformer()
        
        if background_image is None:
            canvas = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            canvas[:] = self.COLORS['pitch_fill']
        else:
            canvas = background_image.copy()
        
        # 不裁剪投影（用于球员、球，避免头部被裁剪导致比例异常）
        def project(p):
            cam_pts = transformer.world_to_camera([p], camera_params)
            img_pts = transformer.camera_to_image(cam_pts, camera_params, clip_to_bounds=False)
            return img_pts[0]

        # 不裁剪版投影（用于球场线条，让 cv2 自己做线段裁剪）
        def project_unclipped(p):
            cam_pts = transformer.world_to_camera([p], camera_params)
            img_pts = transformer.camera_to_image(cam_pts, camera_params, clip_to_bounds=False)
            return img_pts[0]

        # 0. 绘制地面网格
        if show_ground_grid:
            self._draw_ground_grid(canvas, project)

        # 1. 绘制球场线条（用不裁剪投影）
        if show_pitch_lines:
            self._draw_pitch_lines(canvas, pitch_key_points, project_unclipped)
        
        # 2. 绘制视野锥 (持球队员)
        if show_view_cone:
            ball_carrier = next((p for p in players if p.is_ball_carrier), None)
            if ball_carrier:
                self._draw_view_cone(canvas, ball_carrier, project, fov_angle)
        
        # 3. 绘制球员
        for player in players:
            self._draw_player(canvas, player, project)

        # 4. 绘制足球（最后绘制，避免被球员遮挡）
        ball_pos = project(ball.position)
        if ball_pos:
            self._draw_ball(canvas, ball, project)
        
        return canvas
    
    def _draw_pitch_lines(self, canvas, pitch_points, project_func):
        """绘制球场线条"""
        try:
            import cv2
        except ImportError:
            return

        def draw_line(p1_world, p2_world, color=None, thickness=2):
            if color is None:
                color = self.COLORS['pitch_lines']
            p1 = project_func(p1_world)
            p2 = project_func(p2_world)
            if p1 is not None and p2 is not None:
                cv2.line(canvas, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), color, thickness)

        def draw_polyline_clipped(points_world, closed=False, color=None, thickness=1):
            """画折线，只连接相邻两端都在相机前方的线段"""
            if color is None:
                color = self.COLORS['pitch_lines']
            for i in range(len(points_world) - 1):
                draw_line(points_world[i], points_world[i+1], color, thickness)
            if closed and len(points_world) >= 2:
                draw_line(points_world[-1], points_world[0], color, thickness)

        def sample_arc(cx, cy, radius, start, end, segments=48):
            return [
                np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), 0.0])
                for a in np.linspace(start, end, segments + 1)
            ]

        def draw_spot(center, radius=0.25, color=None):
            if color is None:
                color = self.COLORS['pitch_lines']
            projected = project_func(np.array([center[0], center[1], 0.0]))
            if projected is None:
                return
            edge = project_func(np.array([center[0] + radius, center[1], 0.0]))
            pixel_radius = 4
            if edge is not None:
                pixel_radius = max(3, int(np.linalg.norm(np.array(edge) - np.array(projected))))
            cv2.circle(canvas, (int(projected[0]), int(projected[1])), pixel_radius, color, -1)

        length = 105.0
        width = 68.0
        half_l = length / 2
        half_w = width / 2
        penalty_l = 16.5
        penalty_hw = 40.32 / 2
        goal_area_l = 5.5
        goal_area_hw = 18.32 / 2
        penalty_spot_x = half_l - 11.0
        penalty_arc_r = 9.15
        goal_hw = 7.32 / 2
        goal_depth = 2.0

        # 外边界
        corners = [
            np.array([-half_l, -half_w, 0.0]),
            np.array([-half_l, half_w, 0.0]),
            np.array([half_l, half_w, 0.0]),
            np.array([half_l, -half_w, 0.0]),
            np.array([-half_l, -half_w, 0.0]),
        ]
        draw_polyline_clipped(corners, thickness=2)

        # 中线
        draw_line(np.array([0.0, -half_w, 0.0]), np.array([0.0, half_w, 0.0]))

        # 两侧大禁区
        left_pa = [
            np.array([-half_l, -penalty_hw, 0.0]),
            np.array([-half_l + penalty_l, -penalty_hw, 0.0]),
            np.array([-half_l + penalty_l, penalty_hw, 0.0]),
            np.array([-half_l, penalty_hw, 0.0]),
        ]
        right_pa = [
            np.array([half_l, -penalty_hw, 0.0]),
            np.array([half_l - penalty_l, -penalty_hw, 0.0]),
            np.array([half_l - penalty_l, penalty_hw, 0.0]),
            np.array([half_l, penalty_hw, 0.0]),
        ]
        draw_polyline_clipped(left_pa, closed=True, thickness=2)
        draw_polyline_clipped(right_pa, closed=True, thickness=2)

        # 两侧小禁区
        left_ga = [
            np.array([-half_l, -goal_area_hw, 0.0]),
            np.array([-half_l + goal_area_l, -goal_area_hw, 0.0]),
            np.array([-half_l + goal_area_l, goal_area_hw, 0.0]),
            np.array([-half_l, goal_area_hw, 0.0]),
        ]
        right_ga = [
            np.array([half_l, -goal_area_hw, 0.0]),
            np.array([half_l - goal_area_l, -goal_area_hw, 0.0]),
            np.array([half_l - goal_area_l, goal_area_hw, 0.0]),
            np.array([half_l, goal_area_hw, 0.0]),
        ]
        draw_polyline_clipped(left_ga, closed=True, thickness=2)
        draw_polyline_clipped(right_ga, closed=True, thickness=2)

        # 球门框、点球点、中圈、禁区弧、角球弧
        draw_polyline_clipped([
            np.array([-half_l, -goal_hw, 0.0]),
            np.array([-half_l - goal_depth, -goal_hw, 0.0]),
            np.array([-half_l - goal_depth, goal_hw, 0.0]),
            np.array([-half_l, goal_hw, 0.0]),
        ], thickness=2)
        draw_polyline_clipped([
            np.array([half_l, -goal_hw, 0.0]),
            np.array([half_l + goal_depth, -goal_hw, 0.0]),
            np.array([half_l + goal_depth, goal_hw, 0.0]),
            np.array([half_l, goal_hw, 0.0]),
        ], thickness=2)
        draw_polyline_clipped(sample_arc(0.0, 0.0, 9.15, 0.0, 2 * np.pi, 96), closed=True, thickness=1)
        draw_polyline_clipped(sample_arc(-penalty_spot_x, 0.0, penalty_arc_r, -0.92, 0.92, 36), thickness=1)
        draw_polyline_clipped(sample_arc(penalty_spot_x, 0.0, penalty_arc_r, np.pi - 0.92, np.pi + 0.92, 36), thickness=1)
        draw_polyline_clipped(sample_arc(-half_l, -half_w, 1.0, 0.0, np.pi / 2, 16), thickness=1)
        draw_polyline_clipped(sample_arc(half_l, -half_w, 1.0, np.pi / 2, np.pi, 16), thickness=1)
        draw_polyline_clipped(sample_arc(half_l, half_w, 1.0, np.pi, np.pi * 1.5, 16), thickness=1)
        draw_polyline_clipped(sample_arc(-half_l, half_w, 1.0, np.pi * 1.5, np.pi * 2, 16), thickness=1)
        draw_spot((0.0, 0.0), 0.18)
        draw_spot((-penalty_spot_x, 0.0), 0.22)
        draw_spot((penalty_spot_x, 0.0), 0.22)

        # 深度参考线（每10米）
        for x in range(-50, 55, 10):
            if x == 0:
                continue
            draw_line(np.array([x, -half_w, 0.0]), np.array([x, half_w, 0.0]),
                      color=(100, 100, 100), thickness=1)
    
    def _draw_player(self, canvas, player, project_func):
        """
        绘制游戏风格的站立人物
        身体 = 矩形躯干 + 圆形头部 + 双腿
        """
        try:
            import cv2
        except ImportError:
            return

        # 投影脚部和头部
        foot_pos = player.position.copy()
        foot_pos[2] = 0
        foot_img = project_func(foot_pos)

        head_pos = player.position.copy()
        head_pos[2] = player.height
        head_img = project_func(head_pos)

        if foot_img is None:
            return

        u_foot, v_foot = int(foot_img[0]), int(foot_img[1])
        H, W = canvas.shape[:2]
        if u_foot < -80 or u_foot > W + 80 or v_foot < -120 or v_foot > H + 80:
            return

        # 颜色
        color = self.COLORS[player.team]
        dark = tuple(max(0, c - 60) for c in color)

        if head_img:
            u_head, v_head = int(head_img[0]), int(head_img[1])
            body_h = max(8, abs(v_foot - v_head))
        else:
            # 兜底：按屏幕位置估算高度，越靠近底部越大
            rel = np.clip(1.0 - (v_foot / max(1.0, float(H))), 0.0, 1.0)
            body_h = int(8 + rel * 42)
            v_head = v_foot - body_h
            u_head = u_foot

        body_h = int(np.clip(body_h, 8, 64))

        # --- 绘制阴影 ---
        shadow_pts = np.array([
            [u_foot - body_h // 3, v_foot],
            [u_foot + body_h // 3, v_foot],
            [u_foot + body_h // 2, v_foot + body_h // 6],
            [u_foot - body_h // 2, v_foot + body_h // 6],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [shadow_pts], (20, 60, 20))

        # --- 绘制腿部 ---
        leg_w = max(2, body_h // 8)
        leg_bottom = v_foot
        leg_top = v_foot - body_h * 3 // 8
        # 左腿
        cv2.line(canvas, (u_foot - leg_w * 2, leg_top),
                 (u_foot - leg_w, leg_bottom), dark, max(2, leg_w))
        # 右腿
        cv2.line(canvas, (u_foot + leg_w * 2, leg_top),
                 (u_foot + leg_w, leg_bottom), dark, max(2, leg_w))

        # --- 绘制躯干 (矩形) ---
        torso_top = v_foot - body_h * 7 // 8
        torso_bottom = v_foot - body_h * 3 // 8
        torso_w = max(3, body_h // 4)
        cv2.rectangle(canvas,
                      (u_foot - torso_w, torso_top),
                      (u_foot + torso_w, torso_bottom),
                      color, -1)
        cv2.rectangle(canvas,
                      (u_foot - torso_w, torso_top),
                      (u_foot + torso_w, torso_bottom),
                      (255, 255, 255), 1)

        # --- 绘制头部 (圆形) ---
        head_r = max(3, body_h // 5)
        head_cy = torso_top - head_r
        cv2.circle(canvas, (u_foot, head_cy), head_r, color, -1)
        cv2.circle(canvas, (u_foot, head_cy), head_r, (255, 255, 255), 1)

        # --- 绘制手臂 ---
        arm_y = torso_top + (torso_bottom - torso_top) // 3
        arm_len = torso_w + max(2, body_h // 6)
        cv2.line(canvas, (u_foot - torso_w, arm_y),
                 (u_foot - arm_len, arm_y + body_h // 5), dark, max(2, leg_w))
        cv2.line(canvas, (u_foot + torso_w, arm_y),
                 (u_foot + arm_len, arm_y + body_h // 5), dark, max(2, leg_w))

        # --- 持球者光环 ---
        if player.is_ball_carrier:
            ring_r = max(12, body_h // 2)
            cv2.circle(canvas, (u_foot, v_foot - body_h // 2), ring_r,
                       self.COLORS['ball_carrier_ring'], 2)

        # --- 编号 (躯干中央) ---
        num_y = (torso_top + torso_bottom) // 2
        font_scale = max(0.3, body_h / 80)
        (tw, th), _ = cv2.getTextSize(str(player.player_id),
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(canvas, str(player.player_id),
                    (u_foot - tw // 2, num_y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(canvas, str(player.player_id),
                    (u_foot - tw // 2, num_y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    
    def _draw_ground_grid(self, canvas, project_func):
        """绘制地面网格线提供深度感"""
        try:
            import cv2
        except ImportError:
            return

        grid_color = (40, 90, 40)
        # 横向线 (沿Y轴, 每5米)
        for y in range(-50, 55, 5):
            pts = []
            for x in range(-55, 60, 5):
                img = project_func(np.array([x, y, 0.0]))
                if img is not None:
                    pts.append((int(img[0]), int(img[1])))
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts)], False, grid_color, 1)

        # 纵向线 (沿X轴, 每5米)
        for x in range(-50, 55, 5):
            pts = []
            for y in range(-55, 60, 5):
                img = project_func(np.array([x, y, 0.0]))
                if img is not None:
                    pts.append((int(img[0]), int(img[1])))
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts)], False, grid_color, 1)

    def _draw_ball(self, canvas, ball, project_func):
        """绘制足球"""
        try:
            import cv2
        except ImportError:
            return
        
        ball_img = project_func(ball.position)
        if ball_img is None:
            return

        u, v = int(ball_img[0]), int(ball_img[1])
        H = canvas.shape[0]
        rel = np.clip(1.0 - (v / max(1.0, float(H))), 0.0, 1.0)
        radius = int(np.clip(3 + rel * 7, 3, 10))
        
        cv2.circle(canvas, (u, v), radius, self.COLORS['ball'], -1)
        cv2.circle(canvas, (u, v), radius, (0, 0, 0), 1)
    
    def _draw_view_cone(self, canvas, player, project_func, fov_angle):
        """
        绘制球员视野锥
        
        扇形表示，类似游戏中的视野显示
        """
        try:
            import cv2
        except ImportError:
            return
        
        foot_pos = player.position.copy()
        foot_pos[2] = 0
        foot_img = project_func(foot_pos)
        
        if foot_img is None:
            return
        
        u, v = int(foot_img[0]), int(foot_img[1])
        
        # 视野锥参数
        cone_length = 300  # 像素
        half_fov = np.radians(fov_angle / 2)
        direction = player.direction
        base_angle = np.arctan2(direction[1], direction[0])
        
        # 生成扇形点
        num_points = 20
        angles = np.linspace(base_angle - half_fov, base_angle + half_fov, num_points)
        
        pts = [(u, v)]
        for angle in angles:
            x = int(u + cone_length * np.cos(angle))
            y = int(v + cone_length * np.sin(angle))
            pts.append((x, y))
        pts.append((u, v))
        
        pts_array = np.array(pts, dtype=np.int32)
        
        # 绘制半透明扇形
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_array], (255, 255, 0))
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
    
    def generate_wireframe_scene(self, camera_params, players: List[Player], ball: Ball,
                                  pitch_key_points: List[np.ndarray],
                                  show_axes: bool = True) -> np.ndarray:
        """
        生成线框视图 (SAOT风格的技术视图)
        
        显示所有关键点的3D连线关系
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("需要安装 opencv-python: pip install opencv-python")
        
        from src.view_transformer import ViewTransformer
        
        transformer = ViewTransformer()
        
        canvas = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        def project(p):
            result = transformer.world_to_image([p], camera_params)
            return result[0] if result else None
        
        # 绘制坐标轴
        if show_axes:
            origin = np.array([0, 0, 0])
            axes_length = 5  # 5米
        
            x_end = origin + np.array([axes_length, 0, 0])
            y_end = origin + np.array([0, axes_length, 0])
            z_end = origin + np.array([0, 0, axes_length])
            
            o = project(origin)
            if o:
                for end, color in [(x_end, (0, 0, 255)), (y_end, (0, 255, 0)), (z_end, (255, 0, 0))]:
                    e = project(end)
                    if e:
                        cv2.line(canvas, (int(o[0]), int(o[1])), (int(e[0]), int(e[1])), color, 3)
        
        # 绘制球员-地面连线 (简化人体)
        for player in players:
            foot = player.position.copy()
            foot[2] = 0
            foot_img = project(foot)
            head = player.position.copy()
            head[2] = player.height
            head_img = project(head)
            
            if foot_img and head_img:
                color = self.COLORS[player.team]
                # 身体线
                cv2.line(canvas, (int(foot_img[0]), int(foot_img[1])),
                        (int(head_img[0]), int(head_img[1])), color, 2)
                # 头部圆
                cv2.circle(canvas, (int(head_img[0]), int(head_img[1])), 
                          5, color, -1)
        
        # 绘制球
        ball_img = project(ball.position)
        if ball_img:
            cv2.circle(canvas, (int(ball_img[0]), int(ball_img[1])), 5, (255, 255, 255), -1)
        
        return canvas
