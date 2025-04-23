import os
import numpy as np
import cv2
from tqdm import tqdm
from fer import FER

def extract_visual_time_series(frames_dir):
    detector = FER(mtcnn=True)
    valence_ts, emotion_data = [], []
    for fname in tqdm(sorted(os.listdir(frames_dir))):
        if not fname.endswith(('.jpg', '.png')): continue
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None: continue
        ems = detector.detect_emotions(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if ems:
            probs = ems[0]["emotions"]
            valence_ts.append(probs["happy"] + probs["surprise"] - \
                sum(probs[e] for e in ["angry","disgust","fear","sad"]))
            emotion_data.append(probs)
        else:
            valence_ts.append(0.0)
            emotion_data.append({"happy":0,"surprise":0,"angry":0,"disgust":0,
                                  "fear":0,"sad":0,"neutral":1})
    return np.array(valence_ts), emotion_data