import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)


def extract_face_mesh_time_series(frames_dir):
    saccade, comfort, frown, jaw, mouth_open = [], [], [], [], []
    prev_fix = None

    print("Extracting face mesh features...")
    for fname in tqdm(sorted(os.listdir(frames_dir))):
        if not fname.endswith(('.jpg', '.png')):
            continue
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            # No face detected - use default values
            saccade.append(0)
            comfort.append(0)
            frown.append(0)
            jaw.append(0)
            mouth_open.append(0)
            continue

        lm = results.multi_face_landmarks[0].landmark

        # Calculate eye fixation point (average of left and right eye)
        fix = np.array([(lm[33].x + lm[263].x)/2, (lm[33].y + lm[263].y)/2])

        # Calculate saccade (eye movement) - distance between current and previous fixation
        current_saccade = np.linalg.norm(
            fix - prev_fix) if prev_fix is not None else 0
        saccade.append(current_saccade)
        prev_fix = fix

        # Calculate frown - vertical distance between eyebrows
        current_frown = abs(lm[65].y - lm[295].y)
        frown.append(current_frown)

        # Calculate jaw tension - vertical distance between jaw points
        current_jaw = abs(lm[152].y - lm[10].y)
        jaw.append(current_jaw)

        # Calculate mouth openness - vertical distance between lips
        current_mouth_open = abs(lm[13].y - lm[14].y)
        mouth_open.append(current_mouth_open)

        # Calculate comfort score based on facial features
        # This will be replaced by the weighted fusion in crossmodal_analysis.py
        provisional_comfort = current_mouth_open - \
            (current_frown + current_jaw)
        comfort.append(provisional_comfort)

    # Normalize values to 0-1 range for better interpretation
    # Clipping removes outliers
    if len(saccade) > 0:
        saccade_array = np.array(saccade)
        saccade_array = np.clip(
            saccade_array / np.percentile(saccade_array, 95), 0, 1)

        frown_array = np.array(frown)
        frown_array = np.clip(
            frown_array / np.percentile(frown_array, 95), 0, 1)

        jaw_array = np.array(jaw)
        jaw_array = np.clip(jaw_array / np.percentile(jaw_array, 95), 0, 1)

        mouth_open_array = np.array(mouth_open)
        mouth_open_array = np.clip(
            mouth_open_array / np.percentile(mouth_open_array, 95), 0, 1)

        comfort_array = np.array(comfort)
        comfort_min = np.percentile(comfort_array, 5)
        comfort_max = np.percentile(comfort_array, 95)
        comfort_range = comfort_max - comfort_min
        comfort_array = np.clip(
            (comfort_array - comfort_min) / comfort_range if comfort_range > 0 else 0.5, 0, 1)

        # Print some statistics for debugging
        print(f"Face Mesh Statistics:")
        print(
            f"  Saccade: min={np.min(saccade_array):.2f}, max={np.max(saccade_array):.2f}, mean={np.mean(saccade_array):.2f}")
        print(
            f"  Frown: min={np.min(frown_array):.2f}, max={np.max(frown_array):.2f}, mean={np.mean(frown_array):.2f}")
        print(
            f"  Jaw: min={np.min(jaw_array):.2f}, max={np.max(jaw_array):.2f}, mean={np.mean(jaw_array):.2f}")
        print(
            f"  Mouth Open: min={np.min(mouth_open_array):.2f}, max={np.max(mouth_open_array):.2f}, mean={np.mean(mouth_open_array):.2f}")
        print(
            f"  Raw Comfort: min={np.min(comfort_array):.2f}, max={np.max(comfort_array):.2f}, mean={np.mean(comfort_array):.2f}")
    else:
        # Empty arrays, return zeros
        saccade_array = np.array([0])
        frown_array = np.array([0])
        jaw_array = np.array([0])
        mouth_open_array = np.array([0])
        comfort_array = np.array([0])
        print("Warning: No face mesh features extracted!")

    return saccade_array, comfort_array, {
        "saccade": saccade_array,
        "comfort": comfort_array,
        "frown": frown_array,
        "jaw": jaw_array,
        "mouth_open": mouth_open_array
    }
