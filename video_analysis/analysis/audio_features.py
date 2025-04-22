import numpy as np

def extract_audio_features(audio_file):
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