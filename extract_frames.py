"""
从比赛视频中按固定时间间隔抽帧，生成用于标注训练的截图。

示例：
  python3 extract_frames.py input/match.mp4 --interval 10
  python3 extract_frames.py input/match.mp4 --interval 5 --start 60 --end 1800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 10.0,
    start_seconds: float = 0.0,
    end_seconds: Optional[float] = None,
    prefix: Optional[str] = None,
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
) -> dict:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")
    if interval_seconds <= 0:
        raise ValueError("--interval 必须大于 0")
    if start_seconds < 0:
        raise ValueError("--start 不能小于 0")
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("--end 必须大于 --start")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None

    actual_end = end_seconds
    if actual_end is None:
        actual_end = duration
    if actual_end is None:
        raise RuntimeError("无法读取视频时长，请显式传入 --end")

    stem = prefix or video.stem
    rows = []
    saved = 0
    current = start_seconds

    write_params = []
    ext = image_ext.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        write_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif ext == "png":
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        raise ValueError("--ext 只支持 jpg/jpeg/png")

    while current <= actual_end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, current * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        timestamp = format_timestamp(current)
        frame_idx = int(round(current * fps)) if fps > 0 else saved
        filename = f"{stem}_{timestamp}_f{frame_idx:08d}.{ext}"
        path = out_dir / filename
        cv2.imwrite(str(path), frame, write_params)

        rows.append({
            "file": str(path),
            "video": str(video),
            "time_seconds": round(float(current), 3),
            "timestamp": timestamp,
            "frame_index": frame_idx,
        })
        saved += 1
        current += interval_seconds

    cap.release()

    manifest = {
        "video": str(video),
        "output_dir": str(out_dir),
        "interval_seconds": interval_seconds,
        "start_seconds": start_seconds,
        "end_seconds": actual_end,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "saved_frames": saved,
        "frames": rows,
    }
    manifest_path = out_dir / "frames_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="按固定时间间隔从足球比赛视频抽帧")
    parser.add_argument("video", help="输入视频路径，例如 input/match.mp4")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 data/frames/<视频名>")
    parser.add_argument("--interval", type=float, default=10.0, help="抽帧间隔秒数，默认 10")
    parser.add_argument("--start", type=float, default=0.0, help="起始秒数，默认 0")
    parser.add_argument("--end", type=float, default=None, help="结束秒数，默认视频结束")
    parser.add_argument("--prefix", default=None, help="输出文件名前缀，默认视频文件名")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "jpeg", "png"], help="输出图片格式")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG 质量，默认 95")
    args = parser.parse_args()

    video = Path(args.video)
    output_dir = args.output_dir or str(Path("data/frames") / video.stem)
    manifest = extract_frames(
        video_path=str(video),
        output_dir=output_dir,
        interval_seconds=args.interval,
        start_seconds=args.start,
        end_seconds=args.end,
        prefix=args.prefix,
        image_ext=args.ext,
        jpeg_quality=args.jpeg_quality,
    )

    print("抽帧完成")
    print(f"  视频: {manifest['video']}")
    print(f"  分辨率: {manifest['width']}x{manifest['height']}")
    print(f"  FPS: {manifest['fps']:.3f}")
    if manifest["duration_seconds"] is not None:
        print(f"  时长: {manifest['duration_seconds']:.1f}s")
    print(f"  间隔: {manifest['interval_seconds']}s")
    print(f"  保存帧数: {manifest['saved_frames']}")
    print(f"  输出目录: {manifest['output_dir']}")
    print(f"  清单: {Path(manifest['output_dir']) / 'frames_manifest.json'}")


if __name__ == "__main__":
    main()
