import os
import argparse
import numpy as np
import json
import shutil

from utils.video_io import extract_audio_and_frames
from utils.visualization import create_metrics_visualization
from analysis.text_features import extract_text_features
from analysis.audio_features import extract_audio_features
from analysis.visual_features import extract_visual_time_series
from analysis.face_mesh import extract_face_mesh_time_series
from analysis.crossmodal_analysis import analyze_multimodal_features, calculate_compatibility


def analyze_video(video_path, output_root):
    """
    Analyze a single video and extract multimodal emotion features
    """
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    audio_path = os.path.join(output_dir, "audio.wav")
    frames_dir = os.path.join(output_dir, "frames")

    # 1. Extract frames & audio
    total_frames, fps, duration = extract_audio_and_frames(
        video_path, audio_path, frames_dir)
    print(
        f"\n [{video_name}] {total_frames} frames @ {fps:.2f}fps, duration {duration:.1f}s")

    # 2. Whisper text sentiment
    text, text_feat = extract_text_features(audio_path)

    # Save transcript for later use
    with open(os.path.join(output_dir, "transcript.txt"), "w") as f:
        f.write(text)

    # 3. Audio features (pitch, intensity)
    audio_feat = extract_audio_features(audio_path)

    # 4. Image emotion valence (FER)
    valence, emotion_data = extract_visual_time_series(frames_dir)

    # 5. Face features (mediapipe)
    saccade, comfort, face_features = extract_face_mesh_time_series(frames_dir)

    # 6. Create visualization (chart + CSV)
    metrics_png, metrics_csv = create_metrics_visualization(
        fps, valence, emotion_data, face_features, output_dir
    )

    # 7. GPT cross-modal summary
    analysis = analyze_multimodal_features(
        text, text_feat, audio_feat, valence, saccade, comfort
    )
    with open(os.path.join(output_dir, "crossmodal_analysis.txt"), "w") as f:
        f.write(analysis)

    # 8. Save metrics data as JSON for easier web access
    metrics_data = {
        "text_sentiment": float(text_feat[0]),
        "pitch": float(audio_feat[0]),
        "intensity": float(audio_feat[1]),
        "valence": [float(v) for v in valence],
        "comfort": [float(c) for c in comfort],
        "saccade": [float(s) for s in saccade],
        "emotions": emotion_data
    }

    with open(os.path.join(output_dir, "metrics_data.json"), "w") as f:
        json.dump(metrics_data, f)

    # Copy the source video to the output directory for easy access
    try:
        shutil.copy(video_path, os.path.join(output_dir, "video.mp4"))
    except Exception as e:
        print(f"Warning: Could not copy source video: {e}")

    print(f"\n Analysis complete → {output_dir}")
    return output_dir


def analyze_compatibility(video1_dir, video2_dir, output_root):
    """
    Analyze compatibility between two videos
    """
    print("\n Analyzing compatibility between videos...")

    # Calculate compatibility score and detailed analysis
    score, detailed_analysis = calculate_compatibility(video1_dir, video2_dir)

    # Save results
    compatibility_data = {
        "score": score,
        "detailed_analysis": detailed_analysis
    }

    output_path = os.path.join(output_root, "compatibility_score.json")
    with open(output_path, "w") as f:
        json.dump(compatibility_data, f)

    detailed_path = os.path.join(output_root, "compatibility_analysis.txt")
    with open(detailed_path, "w") as f:
        f.write(detailed_analysis)

    print(f"Compatibility Score: {score}/100")
    return score, detailed_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multimodal emotion analysis on video(s)")
    parser.add_argument("video_paths", nargs="+",
                        help="Path(s) to video file(s)")
    parser.add_argument("--output-dir", default="output",
                        help="Root directory for output")
    args = parser.parse_args()

    # If two videos are provided, analyze both and calculate compatibility
    if len(args.video_paths) == 2:
        video1_dir = analyze_video(args.video_paths[0], args.output_dir)
        video2_dir = analyze_video(args.video_paths[1], args.output_dir)
        analyze_compatibility(video1_dir, video2_dir, args.output_dir)
    else:
        # Otherwise just analyze individual videos
        for video_path in args.video_paths:
            analyze_video(video_path, args.output_dir)
