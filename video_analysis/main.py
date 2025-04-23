import os
import argparse
import numpy as np

from utils.video_io import extract_audio_and_frames
from utils.visualization import create_metrics_visualization
from analysis.text_features import extract_text_features
from analysis.audio_features import extract_audio_features
from analysis.visual_features import extract_visual_time_series
from analysis.face_mesh import extract_face_mesh_time_series
from analysis.crossmodal_analysis import analyze_multimodal_features


def analyze_video(video_path, output_root):
    # 获取视频文件名，不带扩展名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # 设定输出路径
    audio_path = os.path.join(output_dir, "audio.wav")
    frames_dir = os.path.join(output_dir, "frames")

    # 1. 抽帧 & 音频
    total_frames, fps, duration = extract_audio_and_frames(video_path, audio_path, frames_dir)
    print(f"\n🎞️ [{video_name}] {total_frames}帧 @ {fps:.2f}fps, 时长 {duration:.1f}s")

    # 2. Whisper 文本情绪
    text, text_feat = extract_text_features(audio_path)

    # 3. 声音特征（pitch, intensity）
    audio_feat = extract_audio_features(audio_path)

    # 4. 图像情绪 valence（FER）
    valence, emotion_data = extract_visual_time_series(frames_dir)

    # 5. 面部特征（mediapipe）
    saccade, comfort, face_features = extract_face_mesh_time_series(frames_dir)

    # 6. 可视化输出（图 + CSV）
    metrics_png, metrics_csv = create_metrics_visualization(
        fps, valence, emotion_data, face_features, output_dir
    )

    # 7. GPT 跨模态总结
    analysis = analyze_multimodal_features(
        text, text_feat, audio_feat, valence, saccade, comfort
    )
    with open(os.path.join(output_dir, "crossmodal_analysis.txt"), "w") as f:
        f.write(analysis)

    print(f"\n✅ 完成 → {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal emotion analysis on video(s)")
    parser.add_argument("video_paths", nargs="+", help="Path(s) to video file(s)")
    parser.add_argument("--output-dir", default="output", help="Root directory for output")
    args = parser.parse_args()

    for video_path in args.video_paths:
        analyze_video(video_path, args.output_dir)
