import os
import numpy as np
import cv2
from tqdm import tqdm

def extract_visual_time_series(frames_dir):
    """Returns per-frame emotion valence = (happy+surprise)-(angry+disgust+fear+sad)."""
    try:
        from fer import FER

        detector = FER(mtcnn=True)
        print("🧠 Initialized FER detector with MTCNN")

        valence_ts = []
        emotion_data = []
        detected_count = 0

        for fname in tqdm(sorted(os.listdir(frames_dir)), desc="🔍 Analyzing emotions"):
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
            neg = probs["angry"] + probs["disgust"] + probs["fear"] + probs["sad"]
            valence_ts.append(pos - neg)
            emotion_data.append(probs)

        print(f"😊 Processed {len(valence_ts)} frames, detected faces in {detected_count}")

    except Exception as e:
        print(f"⚠️ Error in visual emotion extraction: {e}")
        valence_ts = []
        emotion_data = []

    return np.array(valence_ts), emotion_data
