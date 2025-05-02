import os
import subprocess
import tempfile
from werkzeug.utils import secure_filename


def handle_recorded_video(file, output_path):
    """
    Process a recorded video file (webm format) and convert it to mp4 format

    Args:
        file: The uploaded file object
        output_path: The destination path for the converted file

    Returns:
        str: Path to the converted file
    """
    # Create temp file to store uploaded webm
    temp_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False)
    temp_webm_path = temp_webm.name
    temp_webm.close()

    # Save uploaded file to temp location
    file.save(temp_webm_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert webm to mp4 using ffmpeg
    try:
        subprocess.check_call([
            'ffmpeg',
            '-i', temp_webm_path,
            '-c:v', 'libx264',  # Video codec
            '-c:a', 'aac',       # Audio codec
            '-strict', 'experimental',
            '-b:a', '192k',      # Audio bitrate
            '-y',                # Overwrite output file if it exists
            output_path
        ])
        print(f"Successfully converted {temp_webm_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        # If conversion fails, just copy the original file
        import shutil
        shutil.copy(temp_webm_path, output_path)
        print(f"Copied original file to {output_path}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_webm_path):
            os.unlink(temp_webm_path)

    return output_path


def process_recorded_videos(video1_file, video2_file, upload_folder):
    """
    Process two recorded videos and prepare them for analysis

    Args:
        video1_file: The first uploaded video file
        video2_file: The second uploaded video file
        upload_folder: The base folder for uploaded files

    Returns:
        tuple: Paths to both processed video files
    """
    video1_path = os.path.join(upload_folder, 'video_1.mp4')
    video2_path = os.path.join(upload_folder, 'video_2.mp4')

    # Process each video
    handle_recorded_video(video1_file, video1_path)
    handle_recorded_video(video2_file, video2_path)

    return video1_path, video2_path


def split_screen_recording(input_video, output_folder):
    """
    Split a single recording containing two people (left and right sides) into two separate videos

    Args:
        input_video: Path to the input video with split screen
        output_folder: Folder to save the output videos

    Returns:
        tuple: Paths to the two output videos
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define output paths
    left_video = os.path.join(output_folder, 'video_1.mp4')
    right_video = os.path.join(output_folder, 'video_2.mp4')

    # Calculate the middle point to split the video
    # First, get video dimensions
    dimensions_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        input_video
    ]

    try:
        dimensions = subprocess.check_output(
            dimensions_cmd).decode('utf-8').strip().split(',')
        width = int(dimensions[0])
        height = int(dimensions[1])

        # Split left half
        left_cmd = [
            'ffmpeg',
            '-i', input_video,
            '-filter:v', f'crop={width//2}:{height}:0:0',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            left_video
        ]
        subprocess.check_call(left_cmd)

        # Split right half
        right_cmd = [
            'ffmpeg',
            '-i', input_video,
            '-filter:v', f'crop={width//2}:{height}:{width//2}:0',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            right_video
        ]
        subprocess.check_call(right_cmd)

        print(f"Successfully split video into {left_video} and {right_video}")

        return left_video, right_video

    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error splitting video: {e}")
        return None, None
