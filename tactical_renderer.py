"""
战术风格第一人称视角渲染器
在真实转播画面上叠加战术分析元素
"""

import numpy as np
import cv2
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class TeamSide(Enum):
    """球队阵营"""
    HOME = "home"      # 主队
    AWAY = "away"      # 客队


@dataclass
class Player:
    """球员数据"""
    player_id: int
    position: np.ndarray  # 3D位置 [x, y, 0]
    direction: np.ndarray  # 面向方向
    team: TeamSide
    is_ball_carrier: bool = False
    pixel_pos: Optional[Tuple[float, float]] = None  # 在原始图像中的像素位置


class TacticalRenderer:
    """
    战术风格渲染器
    
    在真实转播画面上叠加战术元素：
    - 球员位置标记（带编号）
    - 持球队员的视野锥
    - 传球路线
    - 球场关键线条
    """
    
    COLORS = {
        TeamSide.HOME: (0, 180, 255),      # 橙色 (荷兰)
        TeamSide.AWAY: (255, 0, 0),        # 蓝色 (阿根廷)
        'ball_carrier': (0, 255, 0),       # 持球者光环 (绿色)
        'view_cone': (255, 255, 0),        # 视野锥 (黄色)
        'pass_line': (255, 255, 255),      # 传球路线 (白色)
        'pitch_line': (255, 255, 255),     # 球场线条 (白色)
        'text': (255, 255, 255),           # 文字 (白色)
        'shadow': (0, 0, 0),               # 阴影 (黑色)
    }
    
    def __init__(self, image_size: Tuple[int, int] = (1558, 602)):
        self.image_size = image_size
    
    def render_tactical_view(self, 
                             background_image: np.ndarray,
                             camera_params,
                             players: List[Player],
                             ball_carrier: Player,
                             show_view_cone: bool = True,
                             show_pass_lines: bool = True,
                             show_pitch_lines: bool = False) -> np.ndarray:
        """
        渲染战术风格视图
        
        Args:
            background_image: 原始转播画面
            camera_params: 相机参数
            players: 所有球员
            ball_carrier: 持球队员
            show_view_cone: 显示视野锥
            show_pass_lines: 显示传球路线
            show_pitch_lines: 显示球场线条
            
        Returns:
            渲染后的图像
        """
        from src.view_transformer import ViewTransformer
        
        transformer = ViewTransformer()
        canvas = background_image.copy()
        
        def project(p):
            """投影3D点到图像坐标"""
            result = transformer.world_to_image([p], camera_params)
            return result[0] if result else None
        
        # 1. 绘制球场关键线条（可选）
        if show_pitch_lines:
            self._draw_pitch_lines(canvas, project)
        
        # 2. 绘制传球路线
        if show_pass_lines:
            self._draw_pass_lines(canvas, players, ball_carrier, project)
        
        # 3. 绘制视野锥
        if show_view_cone:
            self._draw_view_cone(canvas, ball_carrier, project)
        
        # 4. 绘制球员标记
        self._draw_players(canvas, players, ball_carrier, project)
        
        # 5. 添加信息栏
        self._draw_info_bar(canvas, players, ball_carrier)
        
        return canvas
    
    def _draw_pitch_lines(self, canvas, project_func):
        """绘制球场关键线条"""
        try:
            import cv2
        except ImportError:
            return
        
        # 简化的球场关键点
        pitch_lines = [
            # 中线
            ([0, 34, 0], [0, -34, 0]),
            # 左禁区线
            ([-36, 20.16, 0], [-36, -20.16, 0]),
            # 左球门线
            ([-52.5, 3.66, 0], [-52.5, -3.66, 0]),
        ]
        
        for p1_world, p2_world in pitch_lines:
            p1 = project_func(np.array(p1_world))
            p2 = project_func(np.array(p2_world))
            if p1 and p2:
                cv2.line(canvas, 
                        (int(p1[0]), int(p1[1])), 
                        (int(p2[0]), int(p2[1])), 
                        self.COLORS['pitch_line'], 1)
    
    def _draw_view_cone(self, canvas, ball_carrier, project_func):
        """绘制持球队员的视野锥"""
        try:
            import cv2
        except ImportError:
            return
        
        foot_pos = project_func(ball_carrier.position)
        if foot_pos is None:
            return
        
        u, v = int(foot_pos[0]), int(foot_pos[1])
        
        # 视野锥参数
        cone_length_pixels = 250  # 像素长度
        fov_angle = 60.0  # 度
        half_fov = np.radians(fov_angle / 2)
        
        # 球员面向方向
        direction = ball_carrier.direction
        base_angle = np.arctan2(direction[1], direction[0])
        
        # 生成扇形点
        num_points = 30
        angles = np.linspace(base_angle - half_fov, base_angle + half_fov, num_points)
        
        pts = [(u, v)]
        for angle in angles:
            # 注意：Y轴向下，所以需要反转
            x = int(u + cone_length_pixels * np.cos(angle))
            y = int(v + cone_length_pixels * np.sin(angle))
            pts.append((x, y))
        pts.append((u, v))
        
        pts_array = np.array(pts, dtype=np.int32)
        
        # 绘制半透明扇形
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_array], (0, 255, 255))  # 黄色
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        
        # 绘制边界线
        left_angle = base_angle - half_fov
        right_angle = base_angle + half_fov
        
        left_x = int(u + cone_length_pixels * np.cos(left_angle))
        left_y = int(v + cone_length_pixels * np.sin(left_angle))
        right_x = int(u + cone_length_pixels * np.cos(right_angle))
        right_y = int(v + cone_length_pixels * np.sin(right_angle))
        
        cv2.line(canvas, (u, v), (left_x, left_y), (255, 255, 0), 2)
        cv2.line(canvas, (u, v), (right_x, right_y), (255, 255, 0), 2)
        
        # 绘制弧线
        arc_pts = []
        for angle in angles:
            x = int(u + cone_length_pixels * np.cos(angle))
            y = int(v + cone_length_pixels * np.sin(angle))
            arc_pts.append([x, y])
        arc_pts = np.array(arc_pts, dtype=np.int32)
        cv2.polylines(canvas, [arc_pts], False, (255, 255, 0), 2)
    
    def _draw_pass_lines(self, canvas, players, ball_carrier, project_func):
        """绘制传球路线（持球队员到队友的连线）"""
        try:
            import cv2
        except ImportError:
            return
        
        carrier_pos = project_func(ball_carrier.position)
        if carrier_pos is None:
            return
        
        for player in players:
            if player.is_ball_carrier or player.team != ball_carrier.team:
                continue
            
            teammate_pos = project_func(player.position)
            if teammate_pos is None:
                continue
            
            # 判断传球路线是否合理（距离不要太远）
            dist = np.linalg.norm(player.position[:2] - ball_carrier.position[:2])
            if dist > 40:  # 超过40米不画线
                continue
            
            # 绘制虚线
            p1 = (int(carrier_pos[0]), int(carrier_pos[1]))
            p2 = (int(teammate_pos[0]), int(teammate_pos[1]))
            
            # 根据距离调整透明度
            alpha = max(0.3, 1.0 - dist / 40)
            color = (0, 255, 255) if alpha > 0.6 else (200, 200, 200)
            
            # 绘制虚线
            self._draw_dashed_line(canvas, p1, p2, color, 2)
    
    def _draw_dashed_line(self, canvas, p1, p2, color, thickness):
        """绘制虚线"""
        try:
            import cv2
        except ImportError:
            return
        
        x1, y1 = p1
        x2, y2 = p2
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 10:
            return
        
        dash_length = 15
        gap_length = 10
        num_dashes = int(length / (dash_length + gap_length))
        
        for i in range(num_dashes):
            t1 = i * (dash_length + gap_length) / length
            t2 = min((i * (dash_length + gap_length) + dash_length) / length, 1.0)
            
            pt1 = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
            pt2 = (int(x1 + t2 * (x2 - x1)), int(y1 + t2 * (y2 - y1)))
            
            cv2.line(canvas, pt1, pt2, color, thickness)
    
    def _draw_players(self, canvas, players, ball_carrier, project_func):
        """绘制球员标记"""
        try:
            import cv2
        except ImportError:
            return
        
        for player in players:
            foot_pos = project_func(player.position)
            if foot_pos is None:
                continue
            
            u, v = int(foot_pos[0]), int(foot_pos[1])
            
            # 颜色
            if player.is_ball_carrier:
                # 持球者：绿色大圆 + 光环
                color = self.COLORS['ball_carrier']
                radius = 15
                thickness = 3
            elif player.team == ball_carrier.team:
                # 队友：橙色
                color = self.COLORS[TeamSide.HOME]
                radius = 10
                thickness = 2
            else:
                # 对手：蓝色
                color = self.COLORS[TeamSide.AWAY]
                radius = 8
                thickness = 2
            
            # 绘制球员圆点
            cv2.circle(canvas, (u, v), radius, color, -1)
            cv2.circle(canvas, (u, v), radius, (255, 255, 255), thickness)
            
            # 持球者额外光环
            if player.is_ball_carrier:
                cv2.circle(canvas, (u, v), radius + 5, color, 2)
            
            # 绘制编号
            font_scale = 0.6
            text_offset_y = radius + 15
            cv2.putText(canvas, f"#{player.player_id}", 
                       (u - 15, v + text_offset_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (0, 0, 0), 3)  # 阴影
            cv2.putText(canvas, f"#{player.player_id}", 
                       (u - 15, v + text_offset_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       self.COLORS['text'], 2)
    
    def _draw_info_bar(self, canvas, players, ball_carrier):
        """绘制信息栏"""
        try:
            import cv2
        except ImportError:
            return
        
        h, w = canvas.shape[:2]
        
        # 信息栏背景
        bar_height = 40
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0, 180), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        
        # 信息文字（使用英文避免编码问题）
        teammates = [p for p in players if p.team == ball_carrier.team and not p.is_ball_carrier]
        opponents = [p for p in players if p.team != ball_carrier.team]
        
        info_text = f"Ball Carrier: #{ball_carrier.player_id} | Teammates: {len(teammates)} | Opponents: {len(opponents)} | Visible: {len(players)}"
        cv2.putText(canvas, info_text, 
                   (20, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 255), 2)
        
        # 图例
        legend_y = h - bar_height + 5
        # 持球者
        cv2.circle(canvas, (w - 220, legend_y + 8), 6, (0, 255, 0), -1)
        cv2.putText(canvas, "Ball", (w - 205, legend_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 队友
        cv2.circle(canvas, (w - 140, legend_y + 8), 5, (0, 180, 255), -1)
        cv2.putText(canvas, "Team", (w - 125, legend_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 对手
        cv2.circle(canvas, (w - 70, legend_y + 8), 4, (255, 0, 0), -1)
        cv2.putText(canvas, "Opp", (w - 55, legend_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
