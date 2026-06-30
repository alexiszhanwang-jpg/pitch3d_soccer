# Web Renderer

Three.js 低模预览器，消费 Python 侧导出的通用 `scene_graph.json`。

## 运行

在项目根目录执行：

```bash
python3 -m http.server 8787
```

然后打开：

```text
http://localhost:8787/web_renderer/
```

页面会默认尝试加载 `output_real/scene_graph.json`，也可以手动选择任意 scene graph JSON。

## 设计边界

- Python 视觉管线负责识别与导出结构化场景。
- Three.js 只负责渲染，不做识别。
- `scene_graph.json` 是通用中间层，后续 Unity/Godot/Blender 渲染器都应消费同一份数据。
- 坐标转换由 `src/scene_graph.py` 统一维护，避免各引擎重复猜坐标。
