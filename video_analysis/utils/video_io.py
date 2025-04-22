import os
import subprocess
import cv2
import numpy as np

def extract_audio_and_frames(video_file, audio_file, frames_dir):
    """Extracts the audio track and video frames."""
    subprocess.call(
        f"ffmpeg -y -i {video_file} -q:a 0 -map a {audio_file}", shell=True)
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.call(
        f"ffmpeg -y -i {video_file} {frames_dir}/frame_%04d.jpg", shell=True)
    print(f"✅ Extracted audio → {audio_file} and frames → {frames_dir}/")

    # Get video frame count and fps
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()

    return total_frames, fps, duration

def get_frame_times(video_file, n_frames):
    """Returns a list of time stamps (in seconds) for each frame."""
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return np.arange(n_frames) / fps