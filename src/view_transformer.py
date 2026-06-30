"""
透视变换和视角转换引擎
核心3D数学计算：从转播上帝视角到第一人称视角
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CameraParams:
    """
    相机参数
    
    用于描述转播画面的相机位姿
    """
    # 相机在世界坐标系中的位置 (x, y, z)
    position: np.ndarray  # shape: (3,)
    # 相机朝向 (欧拉角, 弧度)
    # yaw: 绕Z轴旋转 (左右), pitch: 绕X轴旋转 (俯仰), roll: 绕Y轴旋转 (翻滚)
    rotation: np.ndarray = None  # shape: (3,) [yaw, pitch, roll]
    # 相机内参
    focal_length: float = 1000.0  # 焦距 (像素)
    principal_point: Tuple[float, float] = (960, 540)  # 主点 (1920x1080画面中心)
    image_size: Tuple[int, int] = (1920, 1080)  # 画面尺寸 (宽, 高)
    # 直接存储的旋转矩阵 (优先于 rotation 字段使用)
    _R_override: np.ndarray = None  # 3x3
    
    def __post_init__(self):
        if self.rotation is None:
            self.rotation = np.array([0.0, 0.0, 0.0])
    
    def get_rotation_matrix(self) -> np.ndarray:
        """获取3x3旋转矩阵 (camera→world, 与欧拉角一致)"""
        # 如果提供了覆盖矩阵，直接使用
        if self._R_override is not None:
            return self._R_override.T  # 存储的是world→cam, 返回cam→world
        
        yaw, pitch, roll = self.rotation
        
        # 绕Z轴旋转 (yaw) - 水平方向转动
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 绕Y轴旋转 (pitch) - 俯仰角
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 绕X轴旋转 (roll) - 翻滚角
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 旋转顺序: yaw(水平) -> pitch(俯仰) -> roll(翻滚)
        return Rz @ Ry @ Rx
    
    def get_projection_matrix(self) -> np.ndarray:
        """获取3x4投影矩阵 P = K [R|t]"""
        R = self.get_rotation_matrix()
        t = -R @ self.position
        
        K = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])
        
        P = np.hstack([R, t.reshape(3, 1)])
        return K @ P
    
    def get_camera_matrix(self) -> np.ndarray:
        """获取3x3相机内参矩阵"""
        return np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])


class ViewTransformer:
    """
    视角转换器
    
    将3D世界坐标从源相机视角转换到目标相机视角
    """
    
    def __init__(self):
        pass
    
    def world_to_camera(self, world_points: List[np.ndarray], camera: CameraParams) -> List[np.ndarray]:
        """
        将世界坐标转换到相机坐标系
        
        Args:
            world_points: 世界坐标点列表
            camera: 源相机参数
            
        Returns:
            相机坐标系中的点列表
        """
        R = camera.get_rotation_matrix()
        R_inv = R.T  # 旋转矩阵的逆是转置
        t = camera.position
        
        camera_points = []
        for p in world_points:
            # p_cam = R^T (p_world - t)
            p_cam = R_inv @ (p - t)
            camera_points.append(p_cam)
        
        return camera_points
    
    def camera_to_image(self, camera_points: List[np.ndarray], camera: CameraParams,
                         clip_to_bounds: bool = True) -> List[Tuple[float, float]]:
        """
        将相机坐标投影到2D图像平面
        
        Args:
            camera_points: 相机坐标系中的点
            camera: 相机参数
            clip_to_bounds: 是否裁剪到图像边界（False时画面外的点也返回坐标）
            
        Returns:
            2D图像坐标 (u, v)，在相机后方的点返回None
        """
        K = camera.get_camera_matrix()
        image_points = []
        
        for p in camera_points:
            if p is None or p[2] <= 0:  # 在相机后方
                image_points.append(None)
                continue
            
            # 投影
            x_norm = p[0] / p[2]
            y_norm = p[1] / p[2]
            
            u = K[0, 0] * x_norm + K[0, 2]
            # 相机坐标系约定 Y 轴向上，而图像坐标 v 轴向下，需取反
            v = K[1, 2] - K[1, 1] * y_norm
            
            if clip_to_bounds:
                # 严格裁剪：超出画面返回None
                if 0 <= u < camera.image_size[0] and 0 <= v < camera.image_size[1]:
                    image_points.append((u, v))
                else:
                    image_points.append(None)
            else:
                # 不裁剪：超出画面也返回坐标（由调用方/cv2处理裁剪）
                image_points.append((u, v))
        
        return image_points
    
    def world_to_image(self, world_points: List[np.ndarray], camera: CameraParams) -> List[Tuple[float, float]]:
        """
        完整的世界坐标到图像坐标投影
        
        Args:
            world_points: 世界坐标点列表
            camera: 相机参数
            
        Returns:
            2D图像坐标列表
        """
        cam_points = self.world_to_camera(world_points, camera)
        return self.camera_to_image(cam_points, camera)
    
    def estimate_camera_from_points(self, image_points: List[Tuple[float, float]], 
                                     world_points: List[np.ndarray],
                                     image_size: Tuple[int, int] = (1920, 1080),
                                     focal_length: Optional[float] = None) -> CameraParams:
        """
        从已知的2D-3D点对估计相机参数 (PnP问题)
        
        Args:
            image_points: 图像中的2D点
            world_points: 对应的3D世界坐标点
            image_size: 图像尺寸
            focal_length: 已知焦距，如果None则估计
            
        Returns:
            估计的相机参数
        """
        # 使用Direct Linear Transform (DLT) 估计投影矩阵
        # 这里简化实现，实际应使用OpenCV的solvePnP
        
        A = []
        for img_p, world_p in zip(image_points, world_points):
            if img_p is None:
                continue
            u, v = img_p
            x, y, z = world_p[0], world_p[1], world_p[2]
            
            # P * X = lambda * x
            # 展开为两个方程
            A.append([-x, -y, -z, -1, 0, 0, 0, 0, u*x, u*y, u*z, u])
            A.append([0, 0, 0, 0, -x, -y, -z, -1, v*x, v*y, v*z, v])
        
        A = np.array(A)
        
        # SVD求解
        U, S, Vt = np.linalg.svd(A)
        P = Vt[-1].reshape(3, 4)
        
        # 分解投影矩阵获取相机参数
        K, R, t, position = self._decompose_projection_matrix(P, focal_length, image_size)
        
        return CameraParams(
            position=position,
            rotation=np.array([0, 0, 0]),  # 简化，实际应从R分解
            focal_length=K[0, 0],
            principal_point=(K[0, 2], K[1, 2]),
            image_size=image_size
        )
    
    def _decompose_projection_matrix(self, P: np.ndarray, focal_length: Optional[float], 
                                      image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分解投影矩阵 P = K[R|t]
        
        使用RQ分解
        """
        # 提取K的3x3部分
        M = P[:, :3]
        p4 = P[:, 3]
        
        # 如果提供了焦距，构建K
        if focal_length:
            K = np.array([
                [focal_length, 0, image_size[0] / 2],
                [0, focal_length, image_size[1] / 2],
                [0, 0, 1]
            ])
            K_inv = np.linalg.inv(K)
            RQ = K_inv @ M
            
            # QR分解得到R
            Q, R_mat = np.linalg.qr(RQ)
            R = Q
            t = np.linalg.inv(M) @ p4
            position = -R.T @ t
        else:
            # 估计焦距
            # 简化：假设无畸变，主点在中心
            K = np.eye(3)
            K[0, 2] = image_size[0] / 2
            K[1, 2] = image_size[1] / 2
            
            # 估计焦距 (简化版)
            scale = np.linalg.norm(M[0])
            K[0, 0] = scale
            K[1, 1] = scale
            
            K_inv = np.linalg.inv(K)
            R_approx = K_inv @ M
            
            # 正交化
            R = self._orthogonalize(R_approx)
            t = -R @ np.linalg.pinv(M) @ p4
            position = np.linalg.inv(M) @ p4
        
        return K, R, p4, position
    
    def _orthogonalize(self, M: np.ndarray) -> np.ndarray:
        """正交化矩阵 (使用SVD)"""
        U, S, Vt = np.linalg.svd(M)
        return U @ Vt
    
    def create_first_person_camera(self, player_position: np.ndarray, 
                                     player_direction: np.ndarray,
                                     camera_height: float = 1.6,
                                     fov: float = 90.0,
                                     image_size: Tuple[int, int] = (1920, 1080)) -> CameraParams:
        """
        创建第一人称视角相机参数 (战术视角: 球员身后高处俯视)
        
        Args:
            player_position: 球员在球场上的位置 (x, y, 0)
            player_direction: 球员面向方向 (单位向量)
            camera_height: 相机高度 (米)
            fov: 视野角度 (度)
            image_size: 输出图像尺寸
            
        Returns:
            第一人称相机参数
        """
        # 相机位置：球员身后2米，上方camera_height米
        direction_2d = player_direction[:2].copy()
        norm = np.linalg.norm(direction_2d)
        if norm > 0:
            direction_2d = direction_2d / norm
        else:
            direction_2d = np.array([1.0, 0.0])
        
        behind_distance = 15.0
        cam_pos = player_position.copy().astype(float)
        cam_pos[:2] = cam_pos[:2] - direction_2d * behind_distance  # 身后
        cam_pos[2] = camera_height

        # 把相机限制在球场范围内（边线内侧 2 米，避免飞出球场）
        cam_pos[0] = np.clip(cam_pos[0], -50.5, 50.5)
        cam_pos[1] = np.clip(cam_pos[1], -32.0, 32.0)
        
        # 目标点：球员前方20米处
        target = player_position.copy().astype(float)
        target[:2] = target[:2] + direction_2d * 20.0
        target[2] = 0
        
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # world->camera 旋转矩阵
        R = np.vstack([right, up, forward])
        
        # 根据FOV计算焦距
        f = image_size[0] / (2 * np.tan(np.radians(fov / 2)))
        
        params = CameraParams(
            position=cam_pos,
            rotation=np.array([0.0, 0.0, 0.0]),
            focal_length=f,
            principal_point=(image_size[0] / 2, image_size[1] / 2),
            image_size=image_size
        )
        params._R_override = R
        return params
    
    def transform_scene(self, world_points: List[np.ndarray],
                        source_camera: CameraParams,
                        target_camera: CameraParams) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        将整个场景从源相机视角转换到目标相机视角
        
        Args:
            world_points: 场景中所有点的3D世界坐标
            source_camera: 源相机 (转播画面相机)
            target_camera: 目标相机 (第一人称相机)
            
        Returns:
            (源图像坐标, 目标图像坐标)
        """
        source_image = self.world_to_image(world_points, source_camera)
        target_image = self.world_to_image(world_points, target_camera)
        
        return source_image, target_image
    
    def compute_homography(self, src_points: List[Tuple[float, float]], 
                           dst_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        计算单应性矩阵 (用于图像级透视变换)
        
        Args:
            src_points: 源图像中的点
            dst_points: 目标图像中的对应点
            
        Returns:
            3x3 单应性矩阵
        """
        A = []
        for (x1, y1), (x2, y2) in zip(src_points, dst_points):
            A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
        
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]  # 归一化
        
        return H
