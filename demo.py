"""
Demo脚本：展示足球转播视角转换功能

运行方式:
    python demo.py

依赖:
    pip install numpy opencv-python matplotlib
"""

import numpy as np
import os

from src.pitch_3d import Pitch3D
from src.player_renderer import Player, Ball, TeamSide
from src.first_person_view import FirstPersonViewConverter, BroadcastFrame
from src.view_transformer import CameraParams


def create_sample_scene():
    """
    创建一个示例场景：模拟一场正在进行的比赛
    
    假设主队 (蓝色) 从左向右进攻，球在中场附近
    """
    pitch = Pitch3D()
    L = pitch.dim.length / 2  # 半场长度
    W = pitch.dim.width / 2   # 半场宽度
    
    # 创建球员 (22名)
    players = []
    
    # 主队 (蓝色) - 4-3-3 阵型，从左向右进攻
    # 持球队员在中场附近
    home_positions = [
        # 门将
        [-L + 2, 0],
        # 后卫
        [-L + 15, -W + 5], [-L + 15, -8], [-L + 15, 8], [-L + 15, W - 5],
        # 中场
        [-10, -15], [-5, -5], [0, 0],  # 8号持球
        # 前锋
        [10, -15], [20, -5], [10, 15],
    ]
    
    for i, (x, y) in enumerate(home_positions):
        direction = np.array([1, 0, 0])  # 向右进攻
        players.append(Player(
            player_id=i + 1,
            position=np.array([x, y, 0]),
            direction=direction,
            team=TeamSide.HOME,
            is_ball_carrier=(i == 7)  # 8号中场持球 (index 7)
        ))
    
    # 客队 (红色) - 4-4-2 阵型，防守
    away_positions = [
        # 门将
        [L - 2, 0],
        # 后卫
        [L - 15, -W + 5], [L - 15, -8], [L - 15, 8], [L - 15, W - 5],
        # 中场
        [10, -12], [5, -6], [5, 6], [10, 12],
        # 前锋
        [-8, -10], [-8, 10],
    ]
    
    for i, (x, y) in enumerate(away_positions):
        direction = np.array([-1, 0, 0])  # 向左防守
        players.append(Player(
            player_id=i + 12,
            position=np.array([x, y, 0]),
            direction=direction,
            team=TeamSide.AWAY
        ))
    
    # 持球位置 (主队8号)
    ball_carrier_pos = np.array([0, 0, 0])  # 中场
    ball = Ball(position=ball_carrier_pos + np.array([0.5, 0.2, 0.05]))
    
    return pitch, players, ball


def create_broadcast_camera(pitch_obj):
    """
    创建转播塔相机参数 (典型上帝视角)
    
    转播塔通常在球场一侧高处，俯瞰球场
    """
    W = pitch_obj.dim.width / 2
    
    # 转播塔位置：球场一侧边线外高处
    cam_pos = np.array([0, -W - 30, 40])
    
    # 目标点：球场中心
    target = np.array([0, 0, 0])
    
    # 构建 look-at 旋转矩阵 (world->camera)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    # 转播塔视角：右向沿球场长度方向 (X轴)
    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # world->camera 旋转矩阵
    R_w2c = np.vstack([right, up, forward])
    
    cam = CameraParams(
        position=cam_pos,
        rotation=np.array([0, 0, 0]),
        focal_length=1000,
        principal_point=(960, 540),
        image_size=(1920, 1080)
    )
    cam._R_override = R_w2c
    return cam


def main():
    print("=" * 60)
    print("足球转播视角转换 Demo")
    print("=" * 60)
    
    # 1. 创建示例场景
    print("\n[1] 创建示例场景...")
    pitch, players, ball = create_sample_scene()
    
    ball_carrier = next(p for p in players if p.is_ball_carrier)
    print(f"   持球队员: #{ball_carrier.player_id} (主队)")
    print(f"   位置: ({ball_carrier.position[0]:.1f}, {ball_carrier.position[1]:.1f})")
    
    # 2. 创建转播相机
    print("\n[2] 创建转播塔相机...")
    broadcast_camera = create_broadcast_camera(pitch)
    print(f"   位置: ({broadcast_camera.position[0]:.1f}, {broadcast_camera.position[1]:.1f}, {broadcast_camera.position[2]:.1f})")
    
    # 3. 创建转换器
    print("\n[3] 初始化视角转换器...")
    converter = FirstPersonViewConverter(
        pitch=pitch,
        output_size=(1920, 1080),
        first_person_fov=90.0,
        camera_height=1.6  # 球员眼睛高度
    )
    
    # 4. 创建转播帧
    frame = BroadcastFrame(
        source_camera=broadcast_camera,
        players=players,
        ball=ball
    )
    
    # 5. 执行视角转换
    print("\n[4] 执行视角转换...")
    result = converter.convert(frame, show_view_cone=True, show_wireframe=True)
    
    print(f"   第一人称相机位置: ({result.fp_camera.position[0]:.1f}, {result.fp_camera.position[1]:.1f}, {result.fp_camera.position[2]:.1f})")
    print(f"   可见球员数: {len(result.visible_players)}")
    
    # 6. 计算相对位置
    print("\n[5] 计算球员相对位置...")
    rel_positions = converter.get_player_relative_positions(ball_carrier, players)
    
    print("\n   最近的5名球员:")
    sorted_players = sorted(rel_positions, key=lambda x: x['distance'])[:5]
    for rp in sorted_players:
        team_name = "主队" if rp['team'] == TeamSide.HOME else "客队"
        angle_deg = np.degrees(rp['angle'])
        print(f"   - #{rp['player_id']} ({team_name}): {rp['distance']:.1f}m, 角度: {angle_deg:.1f}°")
    
    # 7. 保存结果图像
    print("\n[6] 保存结果图像...")
    
    try:
        import cv2
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存第一人称视图
        fp_path = os.path.join(output_dir, "first_person_view.png")
        cv2.imwrite(fp_path, result.first_person_image)
        print(f"   第一人称视图: {fp_path}")
        
        # 保存线框视图
        if result.wireframe_image is not None:
            wf_path = os.path.join(output_dir, "wireframe_view.png")
            cv2.imwrite(wf_path, result.wireframe_image)
            print(f"   线框视图: {wf_path}")
        
        # 保存俯视图 (top-down)
        top_down_camera = CameraParams(
            position=np.array([0, 0, 60]),
            rotation=np.array([0, 0, 0]),
            focal_length=800,
            principal_point=(960, 540),
            image_size=(1920, 1080)
        )
        # 俯视: 相机向下看，x轴向右，y轴向下(对应世界-y)
        R_td = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        top_down_camera._R_override = R_td
        
        top_down_image = converter.renderer.render_scene(
            camera_params=top_down_camera,
            players=players,
            ball=ball,
            pitch_key_points=pitch.get_pitch_key_points(),
            show_view_cone=False
        )
        td_path = os.path.join(output_dir, "top_down_view.png")
        cv2.imwrite(td_path, top_down_image)
        print(f"   俯视图: {td_path}")
        
        # 保存转播上帝视角
        broadcast_image = converter.renderer.render_scene(
            camera_params=frame.source_camera,
            players=players,
            ball=ball,
            pitch_key_points=pitch.get_pitch_key_points(),
            show_view_cone=False
        )
        bc_path = os.path.join(output_dir, "broadcast_view.png")
        cv2.imwrite(bc_path, broadcast_image)
        print(f"   转播视角: {bc_path}")
        
        print("\n" + "=" * 60)
        print("Demo 完成! 查看 output/ 目录中的图像")
        print("=" * 60)
        
    except ImportError:
        print("   警告: 未安装 opencv-python，无法保存图像")
        print("   安装: pip install opencv-python")
    
    return result


if __name__ == "__main__":
    main()
