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

def run_full_analysis(video_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio.wav")
    frames_dir = os.path.join(output_dir, "frames")

    # Step 1: Extract frames and audio
    total_frames, fps, duration = extract_audio_and_frames(video_path, audio_path, frames_dir)
    print(f"\n🎞️ Video Info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")

    # Step 2: Whisper + GPT (text sentiment)
    text, text_feat = extract_text_features(audio_path)
    print(f"\n📝 Transcript Preview:\n{text[:300]}...")
    print(f"📈 Text Sentiment Score: {text_feat[0]:.2f}")

    # Step 3: Audio features (pitch + intensity)
    audio_feat = extract_audio_features(audio_path)
    print(f"🎵 Audio Features - Pitch: {audio_feat[0]:.1f} Hz | Intensity: {audio_feat[1]:.1f} dB")

    # Step 4: Visual valence (FER)
    valence, emotion_data = extract_visual_time_series(frames_dir)
    print(f"📊 Average Valence: {np.mean(valence):.2f}")

    # Step 5: Facial mesh features
    saccade, comfort, face_features = extract_face_mesh_time_series(frames_dir)
    print(f"🧠 Eye Movement: {np.mean(saccade):.4f} | Comfort: {np.mean(comfort):.4f}")

    # Step 6: Visualization
    fig_path, csv_path = create_metrics_visualization(
        fps, valence, emotion_data, face_features, output_dir
    )

    # Step 7: Cross-modal GPT analysis
    analysis = analyze_multimodal_features(
        text, text_feat, audio_feat, valence, saccade, comfort
    )
    analysis_path = os.path.join(output_dir, "crossmodal_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write(analysis)

    print("\n✅ Full analysis complete!")
    if fig_path:
        print(f"🖼️ Visualization saved to: {fig_path}")
    print(f"📄 GPT Analysis saved to: {analysis_path}")
    print(f"📊 CSV saved to: {csv_path if csv_path else 'N/A'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Video Emotion Analysis Pipeline")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Directory to save output")
    args = parser.parse_args()

    run_full_analysis(args.video_path, args.output_dir)
