import os
import subprocess
import cv2
import numpy as np


def extract_audio_and_frames(video_file, audio_path, frames_dir, target_fps=5):
    """
    Extract audio and frames from video at specific FPS rate

    Args:
        video_file: Path to input video file
        audio_path: Path to save extracted audio
        frames_dir: Directory to save extracted frames
        target_fps: Target frames per second (default: 5)

    Returns:
        tuple: (total frames extracted, original fps, duration in seconds)
    """
    # Extract audio
    subprocess.call(
        f"ffmpeg -y -i {video_file} -q:a 0 -map a {audio_path}", shell=True)

    # Create frames directory
    os.makedirs(frames_dir, exist_ok=True)

    # Get video info
    cap = cv2.VideoCapture(video_file)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_original_frames / original_fps

    # Calculate frame extraction interval to achieve target_fps
    frame_interval = max(1, int(original_fps / target_fps))

    # Extract frames at reduced rate using ffmpeg
    subprocess.call(
        f"ffmpeg -y -i {video_file} -vf \"fps={target_fps}\" {frames_dir}/frame_%04d.jpg",
        shell=True
    )

    # Count how many frames were actually extracted
    extracted_frames = len(
        [f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    print(
        f"Extracted {extracted_frames} frames at {target_fps} FPS from {video_file}")
    print(
        f"Original video: {total_original_frames} frames at {original_fps} FPS, duration: {duration:.2f}s")

    cap.release()

    return extracted_frames, target_fps, duration


def get_frame_times(video_file, n_frames, target_fps=5):
    """Returns a list of time stamps (in seconds) for each frame."""
    return np.arange(n_frames) / target_fps
