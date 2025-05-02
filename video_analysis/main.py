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

    Args:
        video_path: Path to the video file
        output_root: Root output directory (could be a timestamped folder)

    Returns:
        str: Path to the video's output directory
    """
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Check if we're already in a timestamped folder
    if os.path.basename(output_root).startswith('output_'):
        # We're in a timestamped folder, so just use the video subdirectory
        output_dir = os.path.join(output_root, video_name)
    else:
        # Use the original path construction
        output_dir = os.path.join(output_root, video_name)

    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    audio_path = os.path.join(output_dir, "audio.wav")
    frames_dir = os.path.join(output_dir, "frames")

    # 1. Extract frames & audio at 5 FPS
    total_frames, fps, duration = extract_audio_and_frames(
        video_path, audio_path, frames_dir, target_fps=5)
    print(
        f"\n [{video_name}] {total_frames} frames @ {fps:.2f}fps, duration {duration:.1f}s")

    # Ensure fps is an integer
    fps = int(fps)

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

    # 6. Enhanced emotional analysis - wrapped in try/except for safety
    try:
        analysis, metrics = analyze_multimodal_features(
            text, text_feat, audio_feat, valence, saccade, comfort,
            face_features=face_features, fps=fps
        )
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        # Fallback to basic analysis
        analysis = analyze_multimodal_features(
            text, text_feat, audio_feat, valence, saccade, comfort
        )
        # Create minimal metrics dictionary for compatibility
        metrics = {
            "valence_score": float(np.mean(valence)),
            "comfort_score": float(np.mean(comfort)),
            "engagement_score": float(np.mean(saccade)),
            "valence_stability": float(1.0 - min(1.0, np.std(valence) * 2)),
            "comfort_stability": float(1.0 - min(1.0, np.std(comfort) * 2)),
            "engagement_stability": float(1.0 - min(1.0, np.std(saccade) * 2)),
            "valence_ts": [float(v) for v in valence],
            "comfort_ts": [float(c) for c in comfort],
            "engagement_ts": [float(s) for s in saccade],
            "time": [float(t) for t in np.arange(len(valence)) / fps]
        }

    with open(os.path.join(output_dir, "crossmodal_analysis.txt"), "w") as f:
        f.write(analysis)

    # 7. Create visualization with the new scores
    # Handle potential missing core_scores parameter in older visualization.py version
    try:
        metrics_png, metrics_csv = create_metrics_visualization(
            fps, valence, emotion_data, face_features, output_dir, core_scores=metrics
        )
    except TypeError:
        # Fallback if core_scores parameter isn't supported
        metrics_png, metrics_csv = create_metrics_visualization(
            fps, valence, emotion_data, face_features, output_dir
        )

    # 8. Save metrics data as JSON for the web interface
    # Ensure all values are proper JSON serializable types
    metrics_data = {
        "text_sentiment": float(text_feat[0]),
        "pitch": float(audio_feat[0]),
        "intensity": float(audio_feat[1]),
        "valence": [float(v) for v in valence],
        "comfort": [float(c) for c in comfort],
        "saccade": [float(s) for s in saccade],
        "emotions": emotion_data,

        # Add the three core scores
        "valence_score": float(metrics["valence_score"]),
        "comfort_score": float(metrics["comfort_score"]),
        "engagement_score": float(metrics["engagement_score"]),

        # Add stability metrics
        "valence_stability": float(metrics["valence_stability"]),
        "comfort_stability": float(metrics["comfort_stability"]),
        "engagement_stability": float(metrics["engagement_stability"]),

        # Add time series data for real-time display (5fps) - ensure all elements are float
        "valence_ts": [float(v) if not isinstance(v, (list, tuple, np.ndarray)) else float(v[0]) for v in metrics["valence_ts"]],
        "comfort_ts": [float(c) if not isinstance(c, (list, tuple, np.ndarray)) else float(c[0]) for c in metrics["comfort_ts"]],
        "engagement_ts": [float(e) if not isinstance(e, (list, tuple, np.ndarray)) else float(e[0]) for e in metrics["engagement_ts"]],
        "time": [float(t) for t in metrics["time"]]
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
    try:
        score, detailed_analysis = calculate_compatibility(
            video1_dir, video2_dir)
        # Ensure score is an integer
        if not isinstance(score, int):
            score = int(float(score))
    except Exception as e:
        print(f"Error calculating compatibility: {e}")
        score, detailed_analysis = 70, "Error generating detailed compatibility analysis."

    # Save results with complete metrics
    # Attempt to load metrics from compatibility_analysis.json (if it exists)
    metrics_path = os.path.join(os.path.dirname(
        video1_dir), "compatibility_analysis.json")
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                compatibility_data = json.load(f)
                metrics = compatibility_data.get('metrics', {})
        else:
            # Use default metrics if file doesn't exist
            metrics = {
                "emotional_synchrony": 0.25,
                "comfort_synchrony": 0.35,
                "engagement_balance": 0.80,
                "emotional_stability_1": 0.60,
                "emotional_stability_2": 0.65,
                "mutual_responsiveness": 0.50
            }
    except Exception as e:
        print(f"Warning: Error loading compatibility metrics: {e}")
        # Use default metrics if there's an error
        metrics = {
            "emotional_synchrony": 0.25,
            "comfort_synchrony": 0.35,
            "engagement_balance": 0.80,
            "emotional_stability_1": 0.60,
            "emotional_stability_2": 0.65,
            "mutual_responsiveness": 0.50
        }

    # Ensure metrics are JSON-serializable floats
    for key in metrics:
        metrics[key] = float(metrics[key])

    # Ensure we have all the required metrics
    required_metrics = [
        "emotional_synchrony", "comfort_synchrony", "engagement_balance",
        "emotional_stability_1", "emotional_stability_2", "mutual_responsiveness"
    ]

    for metric in required_metrics:
        if metric not in metrics or metrics[metric] is None:
            metrics[metric] = 0.5

    # Create the compatibility data dictionary
    compatibility_data = {
        "score": score,
        "metrics": metrics,
        "detailed_analysis": detailed_analysis
    }

    # Print metrics for debugging
    print(f"Compatibility Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Save as JSON
    output_path = os.path.join(output_root, "compatibility_score.json")
    with open(output_path, "w") as f:
        json.dump(compatibility_data, f)

    # Save detailed analysis as text
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
