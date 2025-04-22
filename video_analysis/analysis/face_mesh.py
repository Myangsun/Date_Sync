# analysis/face_mesh.py

import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

def extract_face_mesh_time_series(frames_dir):
    """Returns saccade + comfort time series and detailed face features."""
    try:
        mpfm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

        saccade_ts, comfort_ts = [], []
        eye_pos_ts = []
        frown_ts, jaw_ts, mouth_open_ts = [], [], []
        prev_fix = None

        for fname in tqdm(sorted(os.listdir(frames_dir)), desc="🔬 Face mesh analysis"):
            if not fname.endswith(('.jpg', '.png')):
                continue

            img = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, fname)), cv2.COLOR_BGR2RGB)
            res = mpfm.process(img)

            if not res.multi_face_landmarks:
                saccade_ts.append(0.0)
                comfort_ts.append(0.0)
                eye_pos_ts.append((0.0, 0.0))
                frown_ts.append(0.0)
                jaw_ts.append(0.0)
                mouth_open_ts.append(0.0)
                continue

            lm = res.multi_face_landmarks[0].landmark
            fix = np.array([(lm[33].x + lm[263].x) / 2, (lm[33].y + lm[263].y) / 2])
            eye_pos_ts.append((fix[0], fix[1]))

            sacc = np.linalg.norm(fix - prev_fix) if prev_fix is not None else 0.0
            prev_fix = fix
            saccade_ts.append(sacc)

            frown = abs(lm[65].y - lm[295].y)
            jaw = abs(lm[152].y - lm[10].y)
            openy = abs(lm[13].y - lm[14].y)

            frown_ts.append(frown)
            jaw_ts.append(jaw)
            mouth_open_ts.append(openy)
            comfort_ts.append(openy - (frown + jaw))

        face_features = {
            "saccade": np.array(saccade_ts),
            "comfort": np.array(comfort_ts),
            "eye_positions": np.array(eye_pos_ts),
            "frown": np.array(frown_ts),
            "jaw": np.array(jaw_ts),
            "mouth_open": np.array(mouth_open_ts),
        }

    except Exception as e:
        print(f"⚠️ Error in face mesh extraction: {e}")
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        face_features = {
            "saccade": np.zeros(frame_count),
            "comfort": np.zeros(frame_count),
            "eye_positions": [(0.5, 0.5)] * frame_count,
            "frown": np.zeros(frame_count),
            "jaw": np.zeros(frame_count),
            "mouth_open": np.zeros(frame_count),
        }

    return face_features["saccade"], face_features["comfort"], face_features

