import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def extract_face_mesh_time_series(frames_dir):
    saccade, comfort, frown, jaw, mouth_open = [], [], [], [], []
    prev_fix = None
    for fname in tqdm(sorted(os.listdir(frames_dir))):
        if not fname.endswith(('.jpg', '.png')): continue
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None: continue
        results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            saccade.append(0); comfort.append(0); frown.append(0); jaw.append(0); mouth_open.append(0)
            continue
        lm = results.multi_face_landmarks[0].landmark
        fix = np.array([(lm[33].x + lm[263].x)/2, (lm[33].y + lm[263].y)/2])
        saccade.append(np.linalg.norm(fix - prev_fix) if prev_fix is not None else 0)
        prev_fix = fix
        frown.append(abs(lm[65].y - lm[295].y))
        jaw.append(abs(lm[152].y - lm[10].y))
        mouth_open.append(abs(lm[13].y - lm[14].y))
        comfort.append(mouth_open[-1] - (frown[-1] + jaw[-1]))
    return np.array(saccade), np.array(comfort), {
        "saccade": np.array(saccade),
        "comfort": np.array(comfort),
        "frown": np.array(frown),
        "jaw": np.array(jaw),
        "mouth_open": np.array(mouth_open)
    }