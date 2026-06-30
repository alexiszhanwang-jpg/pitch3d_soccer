# Pitch3D Soccer

把足球转播截图转换成可交互的低模 3D 站位场景。

当前项目重点是：

1. Python/OpenCV/YOLO 识别球场、球员、足球，并导出 `scene_graph.json`。
2. Three.js 读取 `scene_graph.json`，渲染可拖动的 3D 足球场景。
3. 页面支持上传图片，自动生成场景并展示原图、战术俯视、选中球员跟随和第一人称视角。

> 这不是“从单张图自动复原真实 3D 人体/场景”。当前方案是从图像估计球场平面站位，再用低模 3D 引擎重建战术场景。

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 server.py
```

打开：

```text
http://127.0.0.1:8787/
```

上传一张足球比赛截图后，服务会：

1. 保存图片到 `input/uploads/`
2. 运行视觉识别 pipeline
3. 写出 `output_real/vision_frame.json`
4. 写出 `output_real/scene_graph.json`
5. 在网页中渲染 3D 场景

## Models

本仓库使用 Git LFS 管理 `.pt` 模型文件。克隆后请确保已安装 Git LFS：

```bash
git lfs install
git lfs pull
```

当前默认模型路径：

```text
data/models/football-pitch-detection.pt
runs/detect/runs/detect/football_person_ball_v1_mps/weights/best.pt
```

如果模型不存在，自动识别效果会下降或失败。

## Main Commands

处理单张图片：

```bash
python3 process_real_image.py \
  input/1.jpg \
  --auto-detect \
  --object-model runs/detect/runs/detect/football_person_ball_v1_mps/weights/best.pt \
  --output-dir output_real
```

启动 Web 上传服务：

```bash
python3 server.py
```

运行测试：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

检查前端语法：

```bash
node --check web_renderer/main.js
```

## Repository Layout

```text
server.py                 # Flask 本地上传/API 服务
process_real_image.py     # 单图处理入口，输出调试图和 scene graph
src/
  vision_pipeline.py      # 球场/球员/球识别与像素到世界坐标映射
  scene_graph.py          # 渲染器无关的 JSON 场景图导出
  pitch_3d.py             # 105x68m 球场定义
  player_renderer.py      # OpenCV 静态调试渲染
  view_transformer.py     # 相机与投影数学
web_renderer/
  index.html              # Three.js 页面
  main.js                 # 交互 3D 渲染、上传调用、选中球员视角
  style.css

tests/
  test_server_app.py      # 上传接口最小测试
```

## Web Interaction

- 上传图片：自动生成并渲染 `scene_graph.json`
- 战术俯视：默认可拖动、缩放、平移
- 点击任意球员：设为当前观察对象
- 选中球员跟随：从该球员后方观察
- 选中球员第一人称：从该球员视角观察
- 展示原图：弹层查看当前输入图片

当前交互已经弱化“持球队员识别”依赖：`carrier_id` 只是建议观察对象，没有也可以正常展示站位。

## Outputs

`output_real/` 是运行时输出目录，不提交到 Git：

```text
output_real/scene_graph.json       # Three.js 消费的主场景
output_real/vision_frame.json      # 原始识别证据
output_real/vision_overlay.png     # 检测框/球场关键点叠加
output_real/foot_points_debug.png  # 脚点调试图
output_real/top_down_view.png      # 静态俯视调试图
```

## Coordinate System

- 项目世界坐标：球场中心为原点，`X` 为长度方向，`Y` 为宽度方向，`Z` 为高度。
- Three.js 坐标：`X` 为长度方向，`Y` 为高度，`Z` 为宽度方向的反向映射。
- 标准球场尺寸：`105m x 68m`。

## Notes

- 大视频、抽帧数据、训练数据和运行输出都被 `.gitignore` 排除。
- `output_fifa/`、`data/frames/`、`data/datasets/` 等目录只作为本地实验资产。
- 后续计划可以接入 Roboflow sports/soccer 的 pitch keypoint 配置和 tracking 思路，增强球场定位与视频连续性。
