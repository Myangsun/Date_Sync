from tqdm import tqdm
import os
import numpy as np
import subprocess
import cv2
import tempfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import sys
import moviepy.video.io.VideoFileClip as _vfc
import moviepy.audio.io.AudioFileClip as _afc
import types
# Create a dummy moviepy.editor module to satisfy fer's import
_editor = types.ModuleType("moviepy.editor")
_editor.VideoFileClip = _vfc.VideoFileClip
_editor.AudioFileClip = _afc.AudioFileClip
sys.modules["moviepy.editor"] = _editor

# Initialize OpenAI client
client = OpenAI()


def extract_audio_and_frames(video_file, output_dir="output"):
    """
    Extract audio and frames from the video with timestamps.
    Returns the duration of the video.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Get video info
    video = VideoFileClip(video_file)
    duration = video.duration
    fps = video.fps

    # Extract audio
    audio_file = os.path.join(output_dir, "audio.wav")
    video.audio.write_audiofile(audio_file, logger=None)

    # Extract frames with timestamp information
    timestamps = []
    frame_paths = []

    for i, t in enumerate(tqdm(np.arange(0, duration, 1/fps), desc="Extracting frames")):
        frame = video.get_frame(t)
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        timestamps.append(t)
        frame_paths.append(frame_path)

    video.close()

    return duration, fps, audio_file, frames_dir, timestamps, frame_paths


def segment_audio(audio_file, segment_length=5.0, output_dir="output"):
    """
    Segment the audio file into smaller chunks for processing.
    Returns the paths to the audio segments and their start times.
    """
    segments_dir = os.path.join(output_dir, "audio_segments")
    os.makedirs(segments_dir, exist_ok=True)

    segment_paths = []
    start_times = []

    # Get total duration using ffprobe
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        duration = float(result.stdout.strip())
    except:
        duration = 0.0

    for start_time in np.arange(0, duration, segment_length):
        end_time = min(start_time + segment_length, duration)
        segment_path = os.path.join(
            segments_dir, f"segment_{start_time:.1f}_{end_time:.1f}.wav")

        # Use ffmpeg to extract the segment
        subprocess.call([
            "ffmpeg", "-y", "-i", audio_file,
            "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", segment_path
        ])

        segment_paths.append(segment_path)
        start_times.append(start_time)

    return segment_paths, start_times


def analyze_text_over_time(segment_paths, start_times):
    """
    Transcribe and analyze sentiment for each audio segment over time.
    Returns a DataFrame with time and sentiment data.
    """
    results = []

    for segment_path, start_time in tqdm(zip(segment_paths, start_times), desc="Analyzing text"):
        # Transcribe audio using OpenAI's Whisper API
        with open(segment_path, "rb") as audio_data:
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_data,
                    response_format="text"
                )
                text = transcription
            except Exception as e:
                print(f"Error transcribing segment at {start_time}: {e}")
                text = ""

        # Skip sentiment analysis if no text was transcribed
        if not text.strip():
            results.append({
                "time": start_time,
                "text": "",
                "sentiment": 0.0
            })
            continue

        # Get sentiment analysis using GPT model
        prompt = (
            "Analyze the following transcript for emotional sentiment. "
            "Provide only a single sentiment score between -1 and 1, where -1 is very negative and 1 is very positive. "
            "Return only the numeric score and nothing else.\n\n"
            f"Transcript: {text}"
        )

        try:
            sentiment_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis tool that returns only a single number between -1 and 1."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Extract sentiment score
            sentiment_score = float(
                sentiment_response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error analyzing sentiment at {start_time}: {e}")
            sentiment_score = 0.0

        results.append({
            "time": start_time,
            "text": text,
            "sentiment": sentiment_score
        })

    return pd.DataFrame(results)


def analyze_audio_over_time(segment_paths, start_times):
    """
    Extract pitch and intensity features from each audio segment over time.
    Returns a DataFrame with time and audio feature data.
    """
    import parselmouth

    results = []

    for segment_path, start_time in tqdm(zip(segment_paths, start_times), desc="Analyzing audio"):
        try:
            snd = parselmouth.Sound(segment_path)
            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]

            if len(pitch_values) > 0:
                mean_pitch = pitch_values.mean()
            else:
                mean_pitch = 0

            intensity = snd.to_intensity()
            mean_intensity = intensity.values.mean()

            results.append({
                "time": start_time,
                "pitch": mean_pitch,
                "intensity": mean_intensity
            })
        except Exception as e:
            print(f"Error analyzing audio at {start_time}: {e}")
            results.append({
                "time": start_time,
                "pitch": 0,
                "intensity": 0
            })

    return pd.DataFrame(results)


def analyze_frames_over_time(frame_paths, timestamps, sample_rate=10):
    """
    Extract visual emotions from frames over time at the specified sampling rate.
    Returns a DataFrame with time and visual emotion data.
    """
    from fer import FER

    detector = FER(mtcnn=True)
    results = []

    # Sample frames at the specified rate
    sampled_indices = np.arange(0, len(frame_paths), sample_rate)

    for i in tqdm(sampled_indices, desc="Analyzing frames"):
        frame_path = frame_paths[i]
        timestamp = timestamps[i]

        try:
            # Read and analyze frame
            img = cv2.imread(frame_path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emotions = detector.detect_emotions(rgb)

            if emotions:
                # Get emotions from the first detected face
                emotion_values = emotions[0]["emotions"]

                results.append({
                    "time": timestamp,
                    "angry": emotion_values["angry"],
                    "disgust": emotion_values["disgust"],
                    "fear": emotion_values["fear"],
                    "happy": emotion_values["happy"],
                    "sad": emotion_values["sad"],
                    "surprise": emotion_values["surprise"],
                    "neutral": emotion_values["neutral"]
                })
            else:
                # No face detected
                results.append({
                    "time": timestamp,
                    "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
                    "sad": 0, "surprise": 0, "neutral": 0
                })
        except Exception as e:
            print(f"Error analyzing frame at {timestamp}: {e}")
            results.append({
                "time": timestamp,
                "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
                "sad": 0, "surprise": 0, "neutral": 0
            })

    return pd.DataFrame(results)


def analyze_face_mesh_over_time(frame_paths, timestamps, sample_rate=10):
    """
    Extract face mesh features from frames over time at the specified sampling rate.
    Returns a DataFrame with time and face mesh feature data.
    """
    import mediapipe as mp

    mpfm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = []

    # Sample frames at the specified rate
    sampled_indices = np.arange(0, len(frame_paths), sample_rate)

    for i in tqdm(sampled_indices, desc="Analyzing face mesh"):
        frame_path = frame_paths[i]
        timestamp = timestamps[i]

        try:
            # Read and analyze frame
            img = cv2.imread(frame_path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = mpfm.process(rgb)

            if res.multi_face_landmarks:
                # Get landmarks from the first detected face
                lm = res.multi_face_landmarks[0].landmark

                # Extract features
                left_eye = (lm[33].x, lm[33].y)
                right_eye = (lm[263].x, lm[263].y)
                eye_pos = np.mean([left_eye, right_eye], axis=0)

                frown = abs(lm[65].y - lm[295].y)
                jaw = abs(lm[152].y - lm[10].y)
                mouth_open = abs(lm[13].y - lm[14].y)

                results.append({
                    "time": timestamp,
                    "eye_x": eye_pos[0],
                    "eye_y": eye_pos[1],
                    "frown": frown,
                    "jaw": jaw,
                    "mouth_open": mouth_open
                })
            else:
                # No face detected
                results.append({
                    "time": timestamp,
                    "eye_x": 0, "eye_y": 0,
                    "frown": 0, "jaw": 0, "mouth_open": 0
                })
        except Exception as e:
            print(f"Error analyzing face mesh at {timestamp}: {e}")
            results.append({
                "time": timestamp,
                "eye_x": 0, "eye_y": 0,
                "frown": 0, "jaw": 0, "mouth_open": 0
            })

    return pd.DataFrame(results)


def calculate_eye_fixation(mesh_df):
    """Calculate eye fixation variance over time windows."""
    # Use a rolling window to calculate eye position variance
    eye_positions = mesh_df[['eye_x', 'eye_y']].values

    window_size = 10  # Number of samples to use for variance calculation
    fixation_values = []

    for i in range(len(eye_positions)):
        start_idx = max(0, i - window_size)
        window = eye_positions[start_idx:i+1]
        var_value = np.var(window, axis=0).mean() if len(window) > 1 else 0
        fixation_values.append(var_value)

    mesh_df['eye_fixation_var'] = fixation_values
    return mesh_df


def plot_time_series_results(text_df, audio_df, emotion_df, mesh_df, output_dir="output"):
    """
    Create visualizations of the time series data.
    """
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Plot text sentiment over time
    plt.figure(figsize=(12, 6))
    plt.plot(text_df['time'], text_df['sentiment'],
             marker='o', linestyle='-', color='blue')
    plt.title('Speech Sentiment Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sentiment (-1 to 1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sentiment_over_time.png'), dpi=300)
    plt.close()

    # 2. Plot audio features over time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Pitch (Hz)', color=color)
    ax1.plot(audio_df['time'], audio_df['pitch'],
             marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Intensity (dB)', color=color)
    ax2.plot(audio_df['time'], audio_df['intensity'],
             marker='s', linestyle='-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Voice Pitch and Intensity Over Time')
    fig.tight_layout()
    plt.savefig(os.path.join(
        plots_dir, 'audio_features_over_time.png'), dpi=300)
    plt.close()

    # 3. Plot emotions over time
    plt.figure(figsize=(14, 8))
    emotions = ['happy', 'sad', 'angry', 'fear',
                'disgust', 'surprise', 'neutral']
    colors = ['green', 'blue', 'red', 'purple', 'brown', 'orange', 'gray']

    for emotion, color in zip(emotions, colors):
        plt.plot(emotion_df['time'], emotion_df[emotion], marker='.',
                 linestyle='-', label=emotion.capitalize(), color=color)

    plt.title('Facial Emotions Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'emotions_over_time.png'), dpi=300)
    plt.close()

    # 4. Plot face mesh features over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Eye fixation variance
    ax1.plot(mesh_df['time'], mesh_df['eye_fixation_var'],
             color='purple', marker='.', linestyle='-')
    ax1.set_title('Eye Fixation Variance Over Time')
    ax1.set_ylabel('Variance')
    ax1.grid(True, alpha=0.3)

    # Comfort features
    ax2.plot(mesh_df['time'], mesh_df['frown'], marker='.',
             linestyle='-', label='Frown', color='red')
    ax2.plot(mesh_df['time'], mesh_df['jaw'], marker='.',
             linestyle='-', label='Jaw', color='blue')
    ax2.plot(mesh_df['time'], mesh_df['mouth_open'], marker='.',
             linestyle='-', label='Mouth Open', color='green')
    ax2.set_title('Facial Comfort Features Over Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        plots_dir, 'face_mesh_features_over_time.png'), dpi=300)
    plt.close()

    # 5. Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig)

    # Sentiment
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(text_df['time'], text_df['sentiment'],
             marker='o', linestyle='-', color='blue')
    ax1.set_title('Speech Sentiment')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Sentiment (-1 to 1)')
    ax1.grid(True, alpha=0.3)

    # Audio features
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(audio_df['time'], audio_df['pitch'], marker='o',
             linestyle='-', color='blue', label='Pitch (Hz)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(audio_df['time'], audio_df['intensity'], marker='s',
                  linestyle='-', color='red', label='Intensity (dB)')
    ax2.set_title('Voice Characteristics')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Pitch (Hz)', color='blue')
    ax2_twin.set_ylabel('Intensity (dB)', color='red')

    # Create a combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Positive emotions
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(emotion_df['time'], emotion_df['happy'],
             marker='.', linestyle='-', color='green', label='Happy')
    ax3.plot(emotion_df['time'], emotion_df['surprise'],
             marker='.', linestyle='-', color='orange', label='Surprise')
    ax3.set_title('Positive Emotions')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Probability')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Negative emotions
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(emotion_df['time'], emotion_df['sad'],
             marker='.', linestyle='-', color='blue', label='Sad')
    ax4.plot(emotion_df['time'], emotion_df['angry'],
             marker='.', linestyle='-', color='red', label='Angry')
    ax4.plot(emotion_df['time'], emotion_df['fear'],
             marker='.', linestyle='-', color='purple', label='Fear')
    ax4.plot(emotion_df['time'], emotion_df['disgust'],
             marker='.', linestyle='-', color='brown', label='Disgust')
    ax4.set_title('Negative Emotions')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Probability')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Neutral emotion
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(emotion_df['time'], emotion_df['neutral'],
             marker='.', linestyle='-', color='gray', label='Neutral')
    ax5.set_title('Neutral Emotion')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Probability')
    ax5.grid(True, alpha=0.3)

    # Eye fixation
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(mesh_df['time'], mesh_df['eye_fixation_var'],
             marker='.', linestyle='-', color='purple')
    ax6.set_title('Eye Fixation Variance (Engagement)')
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Variance')
    ax6.grid(True, alpha=0.3)

    # Facial comfort features
    ax7 = fig.add_subplot(gs[3, :])
    ax7.plot(mesh_df['time'], mesh_df['frown'], marker='.',
             linestyle='-', label='Frown', color='red')
    ax7.plot(mesh_df['time'], mesh_df['jaw'], marker='.',
             linestyle='-', label='Jaw', color='blue')
    ax7.plot(mesh_df['time'], mesh_df['mouth_open'], marker='.',
             linestyle='-', label='Mouth Open', color='green')
    ax7.set_title('Facial Comfort Features')
    ax7.set_xlabel('Time (seconds)')
    ax7.set_ylabel('Value')
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'multimodal_dashboard.png'), dpi=300)
    plt.close()

    return os.path.join(plots_dir, 'multimodal_dashboard.png')


def analyze_key_moments(text_df, audio_df, emotion_df, mesh_df):
    """
    Identify and analyze key moments in the video using the time-series data.
    """
    # Identify moments of high emotion (positive or negative)
    high_sentiment_moments = text_df[abs(text_df['sentiment']) > 0.7].copy()

    # Identify moments of high emotional expression
    emotion_intensity = emotion_df[[
        'happy', 'sad', 'angry', 'fear', 'surprise']].max(axis=1)
    high_emotion_indices = emotion_intensity[emotion_intensity > 0.7].index
    high_emotion_moments = emotion_df.iloc[high_emotion_indices].copy()

    # Identify moments of high pitch or intensity variation
    audio_zscore = audio_df.copy()
    audio_zscore['pitch_z'] = (
        audio_df['pitch'] - audio_df['pitch'].mean()) / audio_df['pitch'].std()
    audio_zscore['intensity_z'] = (
        audio_df['intensity'] - audio_df['intensity'].mean()) / audio_df['intensity'].std()
    high_audio_moments = audio_zscore[(abs(audio_zscore['pitch_z']) > 1.5) |
                                      (abs(audio_zscore['intensity_z']) > 1.5)].copy()

    # Combine the key moments
    key_moments = {
        'high_sentiment': high_sentiment_moments,
        'high_emotion': high_emotion_moments,
        'high_audio_variation': high_audio_moments
    }

    return key_moments


def generate_time_based_insights(text_df, audio_df, emotion_df, mesh_df, key_moments):
    """
    Generate insights about the video based on the time-series analysis.
    """
    # Combine the most significant moments for analysis
    significant_times = set()

    # Add high sentiment moments
    for _, row in key_moments['high_sentiment'].iterrows():
        significant_times.add(row['time'])

    # Add high emotion moments
    for _, row in key_moments['high_emotion'].iterrows():
        significant_times.add(row['time'])

    # Add high audio variation moments
    for _, row in key_moments['high_audio_variation'].iterrows():
        significant_times.add(row['time'])

    # Sort the significant times
    significant_times = sorted(list(significant_times))

    # Generate insights for each significant moment
    insights = []

    for time in significant_times:
        # Get the closest data points for each modality
        text_row = text_df.iloc[(text_df['time'] - time).abs().argsort()[0]]
        audio_row = audio_df.iloc[(audio_df['time'] - time).abs().argsort()[0]]
        emotion_row = emotion_df.iloc[(
            emotion_df['time'] - time).abs().argsort()[0]]
        mesh_row = mesh_df.iloc[(mesh_df['time'] - time).abs().argsort()[0]]

        # Prepare the data for analysis
        dominant_emotion = emotion_row[[
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']].idxmax()

        # Create a prompt for OpenAI
        prompt = f"""Analyze this key moment at time {time:.1f} seconds:

1. Speech:
   - Transcript: "{text_row['text']}"
   - Sentiment: {text_row['sentiment']:.2f} (-1 is negative, 1 is positive)

2. Voice:
   - Pitch: {audio_row['pitch']:.1f} Hz
   - Intensity: {audio_row['intensity']:.1f} dB

3. Facial Expression:
   - Dominant emotion: {dominant_emotion}
   - Emotion values: {dict(emotion_row[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']])}

4. Facial Movements:
   - Eye fixation variance: {mesh_row['eye_fixation_var']:.4f}
   - Facial tension (frown): {mesh_row['frown']:.4f}
   - Jaw movement: {mesh_row['jaw']:.4f}
   - Mouth openness: {mesh_row['mouth_open']:.4f}

Provide a brief analysis of this moment (2-3 sentences only), focusing on the alignment or misalignment between different communication channels."""

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in multimodal communication analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )

            insight = completion.choices[0].message.content

            insights.append({
                "time": time,
                "text": text_row['text'],
                "sentiment": text_row['sentiment'],
                "dominant_emotion": dominant_emotion,
                "insight": insight
            })
        except Exception as e:
            print(f"Error generating insight for time {time}: {e}")

    return pd.DataFrame(insights)


def generate_final_report(insights_df, dashboard_path, video_path, output_dir="output"):
    """
    Generate a final report summarizing the analysis results.
    """
    # Create a prompt for the final analysis
    prompt = f"""Generate a comprehensive analysis report for a video communication analysis. Here are key insights from {len(insights_df)} significant moments in the video:

{insights_df[['time', 'text', 'dominant_emotion', 'insight']].to_string(index=False)}

Provide a concise summary (200-300 words) that highlights:
1. Overall communication patterns
2. Key moments of alignment/misalignment between verbal and nonverbal cues
3. Cultural considerations in interpreting these signals
4. Practical suggestions for improving communication based on this analysis"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in communication analysis and cross-cultural understanding."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        report = completion.choices[0].message.content

        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Communication Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .dashboard {{ text-align: center; margin: 30px 0; }}
                .dashboard img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .insights {{ margin: 30px 0; }}
                .insight {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 15px; border-left: 4px solid #3498db; }}
                .time {{ color: #2980b9; font-weight: bold; }}
                .text {{ font-style: italic; color: #555; }}
                .summary {{ background-color: #e8f4f8; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multimodal Communication Analysis Report</h1>
                
                <div class="dashboard">
                    <h2>Communication Dashboard</h2>
                    <img src="{os.path.basename(dashboard_path)}" alt="Communication Dashboard">
                </div>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <div>
                        {report}
                    </div>
                </div>
                
                <div class="insights">
                    <h2>Key Moments Analysis</h2>
        """

        # Add each insight
        for _, row in insights_df.iterrows():
            html_report += f"""
                    <div class="insight">
                        <div class="time">Time: {row['time']:.1f} seconds</div>
                        <div class="text">"{row['text']}"</div>
                        <div>Dominant emotion: {row['dominant_emotion']}</div>
                        <div>Insight: {row['insight']}</div>
                    </div>
            """

        html_report += """
                </div>
            </div>
        </body>
        </html>
        """

        # Save the HTML report
        report_path = os.path.join(output_dir, "analysis_report.html")
        with open(report_path, "w") as f:
            f.write(html_report)

        return report_path, report
    except Exception as e:
        print(f"Error generating final report: {e}")
        return None, "Error generating report."


def analyze_video_over_time(video_path, segment_length=5.0, output_dir="output"):
    """
    Main function to analyze a video over time.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting analysis of video: {video_path}")

    # Step 1: Extract audio and frames with timestamps
    duration, fps, audio_file, frames_dir, timestamps, frame_paths = extract_audio_and_frames(
        video_path, output_dir)
    print(f"Video duration: {duration:.2f} seconds, FPS: {fps}")

    # Step 2: Segment the audio for processing
    segment_paths, start_times = segment_audio(
        audio_file, segment_length, output_dir)
    print(f"Created {len(segment_paths)} audio segments")

    # Step 3: Analyze text over time
    text_df = analyze_text_over_time(segment_paths, start_times)
    print(f"Analyzed text sentiment over {len(text_df)} segments")

    # Step 4: Analyze audio features over time
    audio_df = analyze_audio_over_time(segment_paths, start_times)
    print(f"Analyzed audio features over {len(audio_df)} segments")

    # Step 5: Analyze visual emotions over time
    sample_rate = max(1, int(fps / 2))  # Sample every half second
    emotion_df = analyze_frames_over_time(frame_paths, timestamps, sample_rate)
    print(f"Analyzed visual emotions over {len(emotion_df)} frames")

    # Step 6: Analyze face mesh features over time
    mesh_df = analyze_face_mesh_over_time(frame_paths, timestamps, sample_rate)
    mesh_df = calculate_eye_fixation(mesh_df)
    print(f"Analyzed face mesh features over {len(mesh_df)} frames")

    # Step 7: Plot time series results
    dashboard_path = plot_time_series_results(
        text_df, audio_df, emotion_df, mesh_df, output_dir)
    print(f"Created time series plots at {dashboard_path}")

    # Step 8: Identify and analyze key moments
    key_moments = analyze_key_moments(text_df, audio_df, emotion_df, mesh_df)
    print(
        f"Identified {sum(len(df) for df in key_moments.values())} key moments")

    # Step 9: Generate insights for key moments
    insights_df = generate_time_based_insights(
        text_df, audio_df, emotion_df, mesh_df, key_moments)
    print(f"Generated insights for {len(insights_df)} key moments")

    # Step 10: Generate final report
    report_path, report_text = generate_final_report(
        insights_df, dashboard_path, video_path, output_dir)
    print(f"Generated final report at {report_path}")

    # Return the paths to the generated files and the report text
    results = {
        "dashboard_path": dashboard_path,
        "report_path": report_path,
        "report_text": report_text,
        "data": {
            "text_df": text_df,
            "audio_df": audio_df,
            "emotion_df": emotion_df,
            "mesh_df": mesh_df,
            "insights_df": insights_df
        }
    }

    return results

# Example usage function


def main(video_path):
    """
    Run the full analysis pipeline on a video file.
    """
    output_dir = "output"
    segment_length = 5.0  # Analyze in 5-second segments

    results = analyze_video_over_time(video_path, segment_length, output_dir)

    print("\n" + "="*50)
    print("Analysis complete!")
    print(f"Dashboard saved to: {results['dashboard_path']}")
    print(f"Report saved to: {results['report_path']}")
    print("="*50)
    print("\nExecutive Summary:")
    print(results['report_text'])

    return results


# If running the script directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze video communication over time")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--output-dir", default="output",
                        help="Directory to store output files")
    parser.add_argument("--segment-length", type=float, default=5.0,
                        help="Length of audio segments for analysis (seconds)")

    args = parser.parse_args()

    main(args.video_path)
