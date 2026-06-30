"""
主转换接口
整合所有模块，提供简洁的API进行视角转换
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.pitch_3d import Pitch3D, PitchDimensions
from src.view_transformer import ViewTransformer, CameraParams
from src.player_renderer import PlayerRenderer, Player, Ball, TeamSide


@dataclass
class BroadcastFrame:
    """
    转播画面帧数据
    
    包含一帧转播画面中的所有信息
    """
    # 源转播画面相机参数 (从画面估计或预设)
    source_camera: CameraParams
    
    # 球员列表
    players: List[Player]
    
    # 足球
    ball: Ball
    
    # 可选：图像中的球场关键点 (用于相机标定)
    pitch_image_points: Optional[List[Tuple[float, float]]] = None


@dataclass
class FirstPersonViewResult:
    """
    第一人称视角转换结果
    """
    # 渲染的第一人称图像
    first_person_image: np.ndarray
    
    # 渲染的线框技术视图
    wireframe_image: np.ndarray
    
    # 第一人称相机参数
    fp_camera: CameraParams
    
    # 持球队员
    ball_carrier: Player
    
    # 可见的球员 (在第一人称视野内)
    visible_players: List[Player]


class FirstPersonViewConverter:
    """
    第一人称视角转换器
    
    将转播画面转换为持球队员的第一人称视角
    
    使用流程:
    1. 创建 Pitch3D 对象
    2. 创建转换器实例
    3. 调用 convert() 方法
    """
    
    def __init__(self, pitch: Pitch3D = None, 
                 output_size: Tuple[int, int] = (1920, 1080),
                 first_person_fov: float = 90.0,
                 camera_height: float = 1.6):
        """
        Args:
            pitch: 球场3D对象 (默认使用标准球场)
            output_size: 输出图像尺寸 (宽, 高)
            first_person_fov: 第一人称视野角度 (度)
            camera_height: 第一人称相机高度 (眼睛高度, 米)
        """
        self.pitch = pitch or Pitch3D()
        self.output_size = output_size
        self.first_person_fov = first_person_fov
        self.camera_height = camera_height
        
        self.transformer = ViewTransformer()
        self.renderer = PlayerRenderer(output_size)
    
    def convert(self, frame: BroadcastFrame, 
                ball_carrier_id: Optional[int] = None,
                show_view_cone: bool = True,
                show_wireframe: bool = True) -> FirstPersonViewResult:
        """
        执行视角转换
        
        Args:
            frame: 转播画面帧
            ball_carrier_id: 持球队员ID (如果None则自动找is_ball_carrier=True的球员)
            show_view_cone: 是否显示视野锥
            show_wireframe: 是否生成线框视图
            
        Returns:
            转换结果
        """
        # 1. 确定持球队员
        ball_carrier = self._get_ball_carrier(frame, ball_carrier_id)
        
        # 2. 创建第一人称相机
        fp_camera = self.transformer.create_first_person_camera(
            player_position=ball_carrier.position,
            player_direction=ball_carrier.direction,
            camera_height=self.camera_height,
            fov=self.first_person_fov,
            image_size=self.output_size
        )
        
        # 3. 渲染第一人称视图
        first_person_image = self.renderer.render_scene(
            camera_params=fp_camera,
            players=frame.players,
            ball=frame.ball,
            pitch_key_points=self.pitch.get_pitch_key_points(),
            show_view_cone=show_view_cone,
            fov_angle=self.first_person_fov
        )
        
        # 4. 渲染线框视图 (可选)
        wireframe_image = None
        if show_wireframe:
            wireframe_image = self.renderer.generate_wireframe_scene(
                camera_params=fp_camera,
                players=frame.players,
                ball=frame.ball,
                pitch_key_points=self.pitch.get_pitch_key_points()
            )
        
        # 5. 计算可见球员
        visible_players = self._compute_visible_players(frame.players, fp_camera)
        
        return FirstPersonViewResult(
            first_person_image=first_person_image,
            wireframe_image=wireframe_image,
            fp_camera=fp_camera,
            ball_carrier=ball_carrier,
            visible_players=visible_players
        )
    
    def convert_from_image(self, source_image: np.ndarray,
                           pitch_image_points: List[Tuple[float, float]],
                           players: List[Player],
                           ball: Ball,
                           ball_carrier_id: Optional[int] = None,
                           estimate_camera: bool = True) -> FirstPersonViewResult:
        """
        直接从转播图像和球场关键点进行转换
        
        这是最常用的接口：给定一张转播画面，标定球场关键点，转换视角
        
        Args:
            source_image: 源转播画面 (numpy array)
            pitch_image_points: 画面中球场关键点的2D坐标 (与pitch.get_pitch_key_points()顺序对应)
            players: 球员列表 (世界坐标)
            ball: 足球 (世界坐标)
            ball_carrier_id: 持球队员ID
            estimate_camera: 是否从关键点估计相机参数
            
        Returns:
            转换结果
        """
        # 1. 估计相机参数 (如果需要)
        if estimate_camera:
            world_points = self.pitch.get_pitch_key_points()
            source_camera = self.transformer.estimate_camera_from_points(
                image_points=pitch_image_points,
                world_points=world_points,
                image_size=(source_image.shape[1], source_image.shape[0])
            )
        else:
            # 使用默认相机
            source_camera = CameraParams(
                position=np.array([0, -50, 30]),  # 转播塔典型位置
                rotation=np.array([0, 0.3, 0]),   # 俯视角度
                image_size=(source_image.shape[1], source_image.shape[0])
            )
        
        # 2. 创建帧数据
        frame = BroadcastFrame(
            source_camera=source_camera,
            players=players,
            ball=ball,
            pitch_image_points=pitch_image_points
        )
        
        # 3. 转换
        return self.convert(frame, ball_carrier_id)
    
    def calibrate_camera(self, image_points: List[Tuple[float, float]],
                         world_points: List[np.ndarray] = None,
                         image_size: Tuple[int, int] = (1920, 1080)) -> CameraParams:
        """
        校准相机参数
        
        从画面中的球场关键点估计转播相机的位置和姿态
        
        Args:
            image_points: 图像中的球场关键点2D坐标
            world_points: 对应的3D世界坐标 (默认使用pitch的关键点)
            image_size: 图像尺寸
            
        Returns:
            估计的相机参数
        """
        if world_points is None:
            world_points = self.pitch.get_pitch_key_points()
        
        return self.transformer.estimate_camera_from_points(
            image_points=image_points,
            world_points=world_points,
            image_size=image_size
        )
    
    def _get_ball_carrier(self, frame: BroadcastFrame, ball_carrier_id: Optional[int]) -> Player:
        """获取持球队员"""
        if ball_carrier_id is not None:
            for p in frame.players:
                if p.player_id == ball_carrier_id:
                    return p
            raise ValueError(f"未找到球员ID {ball_carrier_id}")
        
        for p in frame.players:
            if p.is_ball_carrier:
                return p
        
        raise ValueError("未指定持球队员，且没有球员标记为is_ball_carrier=True")
    
    def _compute_visible_players(self, players: List[Player], 
                                  camera: CameraParams) -> List[Player]:
        """
        计算在第一人称视野内可见的球员
        
        使用简单的视锥体裁剪
        """
        visible = []
        
        for player in players:
            # 转换到相机坐标系
            cam_points = self.transformer.world_to_camera([player.position], camera)
            p_cam = cam_points[0]
            
            # 检查是否在相机前方
            if p_cam is not None and p_cam[2] > 0:
                # 检查是否在FOV内 (简化检查)
                fov_half = np.radians(self.first_person_fov / 2)
                angle = np.arctan2(np.sqrt(p_cam[0]**2 + p_cam[1]**2), p_cam[2])
                
                if angle < fov_half + 0.2:  # 加一点余量
                    visible.append(player)
        
        return visible
    
    def get_player_relative_positions(self, ball_carrier: Player, 
                                       other_players: List[Player]) -> List[dict]:
        """
        计算其他球员相对于持球队员的位置信息
        
        返回: 列表，每个元素包含:
        - player_id: 球员ID
        - distance: 距离 (米)
        - angle: 相对角度 (弧度，0为正前方)
        - position: 相对位置向量
        """
        result = []
        carrier_pos = ball_carrier.position
        carrier_dir = ball_carrier.direction
        
        for player in other_players:
            if player.player_id == ball_carrier.player_id:
                continue
            
            # 相对位置向量
            rel_pos = player.position - carrier_pos
            
            # 距离
            distance = np.linalg.norm(rel_pos)
            
            # 相对角度 (在X-Y平面)
            rel_pos_2d = rel_pos[:2]
            carrier_dir_2d = carrier_dir[:2]
            
            angle = np.arctan2(rel_pos_2d[1], rel_pos_2d[0]) - \
                   np.arctan2(carrier_dir_2d[1], carrier_dir_2d[0])
            
            # 归一化到 [-pi, pi]
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            
            result.append({
                'player_id': player.player_id,
                'distance': distance,
                'angle': angle,
                'position': rel_pos,
                'team': player.team
            })
        
        return result
