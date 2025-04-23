import os
import numpy as np
import subprocess
import cv2
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client (make sure your OPENAI_API_KEY is set in env)
client = OpenAI()


def extract_audio_and_frames(video_file, audio_file="audio.wav", frames_dir="frames"):
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


def extract_text_features(audio_file="audio.wav"):
    """Uses Whisper + GPT to get a single sentiment score [-1,1]."""
    # 1) Transcribe
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text"
        )
    text = transcription.strip()
    print("Transcript preview:", text[:100], "...")

    # 2) Ask GPT for a single sentiment score
    prompt = (
        "Provide a single sentiment score between -1 (very negative) and +1 (very positive), "
        "for the following transcript. **Return only that number.**\n\n"
        f"Transcript: {text}"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment-scoring tool."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    try:
        score = float(resp.choices[0].message.content.strip())
    except:
        print("⚠️ Could not parse, defaulting to 0.")
        score = 0.0

    text_feat = np.array([score])
    print("Text feature:", text_feat)
    return text, text_feat


def extract_audio_features(audio_file="audio.wav"):
    """Extracts mean pitch and intensity via praat-parselmouth."""
    try:
        import parselmouth
        snd = parselmouth.Sound(audio_file)
        pitch = snd.to_pitch()
        mean_pitch = pitch.selected_array['frequency'][pitch.selected_array['frequency'] > 0].mean(
        )
        intensity = snd.to_intensity().values.mean()
        audio_feat = np.array([mean_pitch, intensity])
        print("Audio features [Hz, dB]:", audio_feat)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        audio_feat = np.array([220.0, 70.0])  # Default values

    return audio_feat


def extract_visual_time_series(frames_dir="frames"):
    """Returns per-frame emotion valence = (happy+surprise)-(angry+disgust+fear+sad)."""
    try:
        # First, properly import MoviePy if needed for FER
        try:
            import moviepy.editor
        except ImportError:
            # Try to install moviepy if it's not available
            print("Installing moviepy dependency...")
            subprocess.call("pip install moviepy", shell=True)
            import moviepy.editor

        # Now import FER
        from fer import FER

        # Initialize emotion detector with verbose output
        detector = FER(mtcnn=True)
        print("Successfully initialized FER detector")

        valence_ts = []
        emotion_data = []
        frame_count = 0
        detected_count = 0

        for fname in tqdm(sorted(os.listdir(frames_dir)), desc="Analyzing emotions"):
            if not fname.endswith(('.jpg', '.png')):
                continue

            frame_path = os.path.join(frames_dir, fname)
            img = cv2.imread(frame_path)

            if img is None:
                print(f"Warning: Could not read image {frame_path}")
                continue

            frame_count += 1
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Debug
            h, w, _ = rgb_img.shape
            print(f"Processing frame {fname}: size={w}x{h}")

            # Detect emotions
            ems = detector.detect_emotions(rgb_img)

            if not ems:
                print(f"No faces detected in frame {fname}")
                valence_ts.append(0.0)
                emotion_data.append({
                    "happy": 0.0, "surprise": 0.0, "angry": 0.0,
                    "disgust": 0.0, "fear": 0.0, "sad": 0.0, "neutral": 1.0
                })
                continue

            detected_count += 1
            probs = ems[0]["emotions"]
            pos = probs["happy"] + probs["surprise"]
            neg = probs["angry"] + probs["disgust"] + \
                probs["fear"] + probs["sad"]
            valence_ts.append(pos - neg)
            emotion_data.append(probs)

            # Debug
            print(f"Frame {fname}: Emotions = {probs}")

        print(
            f"Processed {frame_count} frames, detected faces in {detected_count} frames")

        if detected_count == 0:
            # Try fallback to Haar Cascade if MTCNN detected no faces
            print("No faces detected with MTCNN, trying with Haar Cascade...")
            detector = FER(mtcnn=False)
            valence_ts = []
            emotion_data = []
            detected_count = 0

            for fname in tqdm(sorted(os.listdir(frames_dir)), desc="Retrying with Haar"):
                if not fname.endswith(('.jpg', '.png')):
                    continue

                frame_path = os.path.join(frames_dir, fname)
                img = cv2.imread(frame_path)

                if img is None:
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ems = detector.detect_emotions(rgb_img)

                if not ems:
                    valence_ts.append(0.0)
                    emotion_data.append({
                        "happy": 0.0, "surprise": 0.0, "angry": 0.0,
                        "disgust": 0.0, "fear": 0.0, "sad": 0.0, "neutral": 1.0
                    })
                    continue

                detected_count += 1
                probs = ems[0]["emotions"]
                pos = probs["happy"] + probs["surprise"]
                neg = probs["angry"] + probs["disgust"] + \
                    probs["fear"] + probs["sad"]
                valence_ts.append(pos - neg)
                emotion_data.append(probs)

            print(f"With Haar: detected faces in {detected_count} frames")

    except Exception as e:
        import traceback
        print(f"Error extracting visual features: {e}")
        print(traceback.format_exc())
        # Count frames in directory
        frame_count = len([f for f in os.listdir(frames_dir)
                          if f.endswith(('.jpg', '.png'))])
        valence_ts = [0.0] * frame_count
        emotion_data = [{"happy": 0.0, "surprise": 0.0, "angry": 0.0, "disgust": 0.0,
                         "fear": 0.0, "sad": 0.0, "neutral": 1.0}] * frame_count

    return np.array(valence_ts), emotion_data


def extract_face_mesh_time_series(frames_dir="frames"):
    """
    Returns two arrays:
      - saccade_ts: per-frame eye movement magnitude
      - comfort_ts: mouth-open minus (frown+jaw-clench)
    """
    try:
        import mediapipe as mp
        mpfm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
        saccade_ts, comfort_ts = [], []
        eye_pos_ts = []
        frown_ts, jaw_ts, mouth_open_ts = [], [], []
        prev_fix = None

        for fname in tqdm(sorted(os.listdir(frames_dir)), desc="Analyzing face mesh"):
            if not fname.endswith(('.jpg', '.png')):
                continue

            img = cv2.cvtColor(cv2.imread(os.path.join(
                frames_dir, fname)), cv2.COLOR_BGR2RGB)
            res = mpfm.process(img)
            if not res.multi_face_landmarks:
                # no face detected
                saccade_ts.append(0.0)
                comfort_ts.append(0.0)
                eye_pos_ts.append((0.0, 0.0))
                frown_ts.append(0.0)
                jaw_ts.append(0.0)
                mouth_open_ts.append(0.0)
                continue

            lm = res.multi_face_landmarks[0].landmark
            # eye fixation point
            fix = np.array([(lm[33].x+lm[263].x)/2, (lm[33].y+lm[263].y)/2])
            eye_pos_ts.append((fix[0], fix[1]))

            if prev_fix is None:
                sacc = 0.0
            else:
                sacc = np.linalg.norm(fix - prev_fix)
            prev_fix = fix
            saccade_ts.append(sacc)

            # comfort: openness minus stress
            frown = abs(lm[65].y - lm[295].y)
            jaw = abs(lm[152].y - lm[10].y)
            openy = abs(lm[13].y - lm[14].y)

            frown_ts.append(frown)
            jaw_ts.append(jaw)
            mouth_open_ts.append(openy)

            comfort_ts.append(openy - (frown + jaw))
    except Exception as e:
        print(f"Error extracting face mesh features: {e}")
        frame_count = len([f for f in os.listdir(frames_dir)
                          if f.endswith(('.jpg', '.png'))])
        saccade_ts = [0.0] * frame_count
        comfort_ts = [0.0] * frame_count
        eye_pos_ts = [(0.5, 0.5)] * frame_count
        frown_ts = [0.0] * frame_count
        jaw_ts = [0.0] * frame_count
        mouth_open_ts = [0.0] * frame_count

    face_features = {
        "saccade": np.array(saccade_ts),
        "comfort": np.array(comfort_ts),
        "eye_positions": np.array(eye_pos_ts),
        "frown": np.array(frown_ts),
        "jaw": np.array(jaw_ts),
        "mouth_open": np.array(mouth_open_ts)
    }

    return np.array(saccade_ts), np.array(comfort_ts), face_features


def get_frame_times(video_file, n_frames):
    """Returns a list of time stamps (in seconds) for each frame."""
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return np.arange(n_frames) / fps


def create_metrics_visualization(video_path, fps, valence, emotion_data, face_features, output_dir="output"):
    """
    Creates a static visualization showing metrics over time that can be used with external tools.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get frame times
    frames = sorted([f for f in os.listdir("frames")
                    if f.endswith(('.jpg', '.png'))])
    times = np.arange(len(frames)) / fps

    # Create Figure
    fig = plt.figure(figsize=(14, 10))

    # Plot metrics
    emotions = ["happy", "sad", "angry", "fear",
                "disgust", "surprise", "neutral"]
    colors = ["green", "blue", "red", "purple", "brown", "orange", "gray"]

    # Plot emotions
    plt.subplot(4, 1, 1)
    for i, emotion in enumerate(emotions):
        emotion_values = [data[emotion] for data in emotion_data]
        plt.plot(times, emotion_values, label=emotion, color=colors[i])
    plt.title("Emotions Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # Plot face features
    plt.subplot(4, 1, 2)
    plt.plot(times, face_features["frown"], label="Frown", color="red")
    plt.plot(times, face_features["jaw"], label="Jaw", color="blue")
    plt.plot(times, face_features["mouth_open"],
             label="Mouth Open", color="green")
    plt.title("Face Features Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # Plot valence
    plt.subplot(4, 1, 3)
    plt.plot(times, valence, label="Emotion Valence", color="blue")
    plt.title("Emotional Valence Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # Plot comfort and saccade
    plt.subplot(4, 1, 4)
    plt.plot(times, face_features["comfort"], label="Comfort", color="green")
    plt.plot(times, face_features["saccade"],
             label="Eye Movement", color="purple")
    plt.title("Comfort & Engagement Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "metrics_over_time.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    # Create a CSV with all metrics for further analysis
    import pandas as pd
    metrics_df = pd.DataFrame({
        'time': times,
        'valence': valence,
        'happy': [data['happy'] for data in emotion_data],
        'sad': [data['sad'] for data in emotion_data],
        'angry': [data['angry'] for data in emotion_data],
        'fear': [data['fear'] for data in emotion_data],
        'disgust': [data['disgust'] for data in emotion_data],
        'surprise': [data['surprise'] for data in emotion_data],
        'neutral': [data['neutral'] for data in emotion_data],
        'frown': face_features['frown'],
        'jaw': face_features['jaw'],
        'mouth_open': face_features['mouth_open'],
        'comfort': face_features['comfort'],
        'saccade': face_features['saccade']
    })

    csv_file = os.path.join(output_dir, "metrics_data.csv")
    metrics_df.to_csv(csv_file, index=False)

    # Create an HTML file with embedded video and metrics for interactive viewing
    html_file = os.path.join(output_dir, "interactive_view.html")

    # Generate frame links
    frame_links = []
    for i, frame in enumerate(frames):
        frame_links.append(f'../frames/{frame}')

    with open(html_file, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; flex-direction: column; max-width: 1200px; margin: 0 auto; }}
                .video-container {{ display: flex; }}
                .video-frame {{ width: 50%; }}
                .metrics {{ width: 50%; }}
                img {{ max-width: 100%; }}
                .controls {{ margin: 10px 0; }}
                input[type=range] {{ width: 100%; }}
                .metrics-container {{ margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
            <script>
                // Frame data
                const frameLinks = {str(frame_links)};
                const frameData = {metrics_df.to_json(orient='records')};
                
                // Current frame index
                let currentFrameIndex = 0;
                
                // Update display based on current frame
                function updateDisplay() {{
                    // Update image
                    document.getElementById('video-frame').src = frameLinks[currentFrameIndex];
                    
                    // Update metrics
                    const data = frameData[currentFrameIndex];
                    document.getElementById('time-value').textContent = data.time.toFixed(2);
                    document.getElementById('valence-value').textContent = data.valence.toFixed(2);
                    document.getElementById('happy-value').textContent = data.happy.toFixed(2);
                    document.getElementById('sad-value').textContent = data.sad.toFixed(2);
                    document.getElementById('angry-value').textContent = data.angry.toFixed(2);
                    document.getElementById('fear-value').textContent = data.fear.toFixed(2);
                    document.getElementById('disgust-value').textContent = data.disgust.toFixed(2);
                    document.getElementById('surprise-value').textContent = data.surprise.toFixed(2);
                    document.getElementById('neutral-value').textContent = data.neutral.toFixed(2);
                    document.getElementById('frown-value').textContent = data.frown.toFixed(4);
                    document.getElementById('jaw-value').textContent = data.jaw.toFixed(4);
                    document.getElementById('mouth-value').textContent = data.mouth_open.toFixed(4);
                    document.getElementById('comfort-value').textContent = data.comfort.toFixed(4);
                    document.getElementById('saccade-value').textContent = data.saccade.toFixed(4);
                    
                    // Update slider
                    document.getElementById('frame-slider').value = currentFrameIndex;
                    
                    // Update frame counter
                    document.getElementById('frame-counter').textContent = 
                        `Frame: ${{currentFrameIndex + 1}} / ${{frameLinks.length}}`;
                }}
                
                // Initialize when page loads
                window.onload = function() {{
                    // Set max value for slider
                    document.getElementById('frame-slider').max = frameLinks.length - 1;
                    
                    // Initial display
                    updateDisplay();
                    
                    // Add event listener for slider
                    document.getElementById('frame-slider').addEventListener('input', function(e) {{
                        currentFrameIndex = parseInt(e.target.value);
                        updateDisplay();
                    }});
                    
                    // Add event listeners for buttons
                    document.getElementById('prev-button').addEventListener('click', function() {{
                        if (currentFrameIndex > 0) {{
                            currentFrameIndex--;
                            updateDisplay();
                        }}
                    }});
                    
                    document.getElementById('next-button').addEventListener('click', function() {{
                        if (currentFrameIndex < frameLinks.length - 1) {{
                            currentFrameIndex++;
                            updateDisplay();
                        }}
                    }});
                    
                    // Add play button functionality
                    let playInterval;
                    document.getElementById('play-button').addEventListener('click', function() {{
                        if (playInterval) {{
                            // Stop playing
                            clearInterval(playInterval);
                            playInterval = null;
                            this.textContent = 'Play';
                        }} else {{
                            // Start playing
                            this.textContent = 'Pause';
                            playInterval = setInterval(function() {{
                                if (currentFrameIndex < frameLinks.length - 1) {{
                                    currentFrameIndex++;
                                    updateDisplay();
                                }} else {{
                                    // End of frames, stop playing
                                    clearInterval(playInterval);
                                    playInterval = null;
                                    document.getElementById('play-button').textContent = 'Play';
                                }}
                            }}, 1000 / 30); // Approximately 30 fps
                        }}
                    }});
                }};
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Video Analysis with Time-Aligned Metrics</h1>
                
                <div class="controls">
                    <button id="prev-button">Previous Frame</button>
                    <button id="play-button">Play</button>
                    <button id="next-button">Next Frame</button>
                    <input type="range" id="frame-slider" min="0" max="0" value="0">
                    <p>Time: <span id="time-value">0.00</span>s | <span id="frame-counter">Frame: 0 / 0</span></p>
                </div>
                
                <div class="video-container">
                    <div class="video-frame">
                        <img id="video-frame" src="" alt="Video Frame">
                    </div>
                    
                    <div class="metrics">
                        <h2>Current Frame Metrics</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Emotion Valence</td>
                                <td id="valence-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Happy</td>
                                <td id="happy-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Sad</td>
                                <td id="sad-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Angry</td>
                                <td id="angry-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Fear</td>
                                <td id="fear-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Disgust</td>
                                <td id="disgust-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Surprise</td>
                                <td id="surprise-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Neutral</td>
                                <td id="neutral-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Frown</td>
                                <td id="frown-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Jaw</td>
                                <td id="jaw-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Mouth Openness</td>
                                <td id="mouth-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Comfort</td>
                                <td id="comfort-value">0.00</td>
                            </tr>
                            <tr>
                                <td>Eye Movement</td>
                                <td id="saccade-value">0.00</td>
                            </tr>
                        </table>
                    </div>
                </div>
                
                <div class="metrics-container">
                    <h2>All Metrics Over Time</h2>
                    <img src="metrics_over_time.png" alt="Metrics Over Time">
                </div>
            </div>
        </body>
        </html>
        """)

    return output_file, csv_file, html_file


def analyze_multimodal_features(text, text_feat, audio_feat, valence, saccade, comfort):
    """Use GPT-4 to produce a cross-cultural insight on fused features."""
    prompt = (
        f"Transcript: \"{text}\"\n\n"
        f"Emotional indicators:\n"
        f"- Text sentiment: {text_feat[0]:.2f}\n"
        f"- Pitch: {audio_feat[0]:.1f} Hz, Intensity: {audio_feat[1]:.1f} dB\n"
        f"- Visual emotion valence: {np.mean(valence):.2f}\n"
        f"- Eye movement: {np.mean(saccade):.4f}\n"
        f"- Comfort level: {np.mean(comfort):.4f}\n\n"
        "Explain how these signals might be interpreted differently across cultures, "
        "and give a concise suggestion for improving communication."
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in cross-cultural nonverbal communication."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    analysis = resp.choices[0].message.content
    print("GPT Analysis:\n", analysis)
    return analysis


def analyze_video(video_file, output_dir="output"):
    """Runs end-to-end multimodal analysis + creates visualization."""
    os.makedirs(output_dir, exist_ok=True)

    # 1) Extract raw modalities
    total_frames, fps, duration = extract_audio_and_frames(video_file)
    print(
        f"Video info: {total_frames} frames, {fps} fps, {duration:.2f}s duration")

    text, text_feat = extract_text_features()
    audio_feat = extract_audio_features()

    # 2) Extract time series data
    valence, emotion_data = extract_visual_time_series()
    saccade, comfort, face_features = extract_face_mesh_time_series()

    # 3) Ask GPT for an overall cultural insight
    analysis = analyze_multimodal_features(
        text, text_feat, audio_feat, valence, saccade, comfort)

    # 4) Create visualization
    output_file, csv_file, html_file = create_metrics_visualization(
        video_file, fps, valence, emotion_data, face_features, output_dir
    )
    print(f"✅ Static visualization saved to: {output_file}")
    print(f"✅ CSV data saved to: {csv_file}")
    print(f"✅ Interactive HTML view saved to: {html_file}")

    # 5) Save analysis to file
    analysis_file = os.path.join(output_dir, "analysis.txt")
    with open(analysis_file, "w") as f:
        f.write(analysis)

    return {
        "visualization": output_file,
        "csv_data": csv_file,
        "interactive_html": html_file,
        "analysis": analysis,
        "analysis_file": analysis_file
    }

# ─────────────────────────────────────────────────────────────────────────────
# Example usage:
# 1) Ensure your OPENAI_API_KEY is set:
#    export OPENAI_API_KEY="sk-..."
# 2) Run:
#    result = analyze_video("your_video.mp4")
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze video with time-aligned metrics visualization")
    parser.add_argument("video_file", help="Path to the video file to analyze")
    parser.add_argument("--output-dir", default="output",
                        help="Directory to store output files")

    args = parser.parse_args()

    result = analyze_video(args.video_file, args.output_dir)
    print(f"\nAnalysis complete!")
    print(f"Interactive visualization: {result['interactive_html']}")
    print(f"Open this HTML file in your browser to view the time-aligned visualization")