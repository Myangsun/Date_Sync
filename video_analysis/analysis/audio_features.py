import numpy as np
import parselmouth

def extract_audio_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch().selected_array['frequency']
    mean_pitch = pitch[pitch > 0].mean()
    intensity = snd.to_intensity().values.mean()
    return np.array([mean_pitch, intensity])