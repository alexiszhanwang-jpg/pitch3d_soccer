"""
球场3D坐标系统
定义标准足球场尺寸和3D坐标变换
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PitchDimensions:
    """标准足球场尺寸 (米)"""
    length: float = 105.0  # 球场长度
    width: float = 68.0    # 球场宽度
    goal_width: float = 7.32   # 球门宽度
    goal_height: float = 2.44  # 球门高度
    penalty_area_length: float = 16.5  # 禁区长度
    penalty_area_width: float = 40.32  # 禁区宽度
    goal_area_length: float = 5.5      # 球门区长度
    goal_area_width: float = 18.32     # 球门区宽度
    center_circle_radius: float = 9.15 # 中圈半径
    penalty_spot_distance: float = 11.0  # 点球点距离


class Pitch3D:
    """
    球场3D坐标系统
    
    坐标系:
    - X轴: 沿球场长度方向 (-length/2 到 +length/2)
    - Y轴: 沿球场宽度方向 (-width/2 到 +width/2)
    - Z轴: 垂直地面方向 (0 为地面)
    
    原点: 球场中心点地面
    """
    
    def __init__(self, dimensions: PitchDimensions = None):
        self.dim = dimensions or PitchDimensions()
        self._cache_key_points()
    
    def _cache_key_points(self):
        """缓存球场关键点坐标"""
        L = self.dim.length / 2
        W = self.dim.width / 2
        pa_l = self.dim.penalty_area_length
        pa_w = self.dim.penalty_area_width / 2
        ga_l = self.dim.goal_area_length
        ga_w = self.dim.goal_area_width / 2
        
        # 球场四角
        self.corners = [
            np.array([-L, -W, 0]),  # 左下
            np.array([-L, W, 0]),   # 左上
            np.array([L, -W, 0]),   # 右下
            np.array([L, W, 0]),    # 右上
        ]
        
        # 禁区角点 (左半场)
        self.left_penalty_corners = [
            np.array([-L + pa_l, -pa_w, 0]),
            np.array([-L + pa_l, pa_w, 0]),
            np.array([-L, -pa_w, 0]),
            np.array([-L, pa_w, 0]),
        ]
        
        # 球门区角点 (左半场)
        self.left_goal_corners = [
            np.array([-L + ga_l, -ga_w, 0]),
            np.array([-L + ga_l, ga_w, 0]),
            np.array([-L, -ga_w, 0]),
            np.array([-L, ga_w, 0]),
        ]
        
        # 球门柱位置 (左半场球门线)
        self.left_goal_posts = [
            np.array([-L, -self.dim.goal_width / 2, 0]),  # 左门柱底部
            np.array([-L, self.dim.goal_width / 2, 0]),   # 右门柱底部
            np.array([-L, -self.dim.goal_width / 2, self.dim.goal_height]),  # 左门柱顶部
            np.array([-L, self.dim.goal_width / 2, self.dim.goal_height]),   # 右门柱顶部
        ]
        
        # 中心点
        self.center = np.array([0, 0, 0])
        
        # 中点 (边线中点)
        self.halfway_top = np.array([0, W, 0])
        self.halfway_bottom = np.array([0, -W, 0])
        
        # 点球点
        self.left_penalty_spot = np.array([-L + self.dim.penalty_spot_distance, 0, 0])
        self.right_penalty_spot = np.array([L - self.dim.penalty_spot_distance, 0, 0])
        
        # 中心圆采样点
        self.center_circle_points = self._sample_circle(0, 0, self.dim.center_circle_radius, 32)
    
    def _sample_circle(self, cx: float, cy: float, radius: float, num_points: int) -> List[np.ndarray]:
        """在X-Y平面上采样圆上的点"""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return [np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), 0]) for a in angles]
    
    def world_to_image(self, points: List[np.ndarray], camera_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """
        将3D世界坐标投影到2D图像坐标
        
        Args:
            points: 3D世界坐标点列表 (x, y, z)
            camera_matrix: 3x4 相机投影矩阵 [K|R|t]
            
        Returns:
            2D图像坐标点列表 (u, v)
        """
        image_points = []
        for p in points:
            # 齐次坐标
            p_h = np.array([p[0], p[1], p[2], 1.0])
            # 投影
            p_proj = camera_matrix @ p_h
            if p_proj[2] > 0:  # 在相机前方
                u = p_proj[0] / p_proj[2]
                v = p_proj[1] / p_proj[2]
                image_points.append((u, v))
            else:
                image_points.append(None)
        return image_points
    
    def get_pitch_key_points(self) -> List[np.ndarray]:
        """获取所有球场关键点 (用于透视标定)"""
        points = []
        points.extend(self.corners)
        points.extend(self.left_penalty_corners)
        points.extend(self.left_goal_corners)
        points.extend(self.center_circle_points)
        points.append(self.center)
        points.append(self.halfway_top)
        points.append(self.halfway_bottom)
        return points
    
    def to_field_coordinates(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        """
        将标准坐标转换为球场坐标
        (0,0) 为球场中心，x正方向为进攻方向
        """
        return np.array([x, y, z])
    
    def get_player_height(self) -> float:
        """球员平均身高 (用于3D表示)"""
        return 1.80
    
    def get_ball_radius(self) -> float:
        """足球半径 (米)"""
        return 0.11
    
    def get_player_radius(self) -> float:
        """球员肩部半径 (米，用于简化表示)"""
        return 0.25
