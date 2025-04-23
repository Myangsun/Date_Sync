import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_metrics_visualization(fps, valence, emotion_data, face_features, output_dir):
    if len(valence) == 0:
        print("⚠️ No valence data detected. Skipping visualization.")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    times = np.arange(len(valence)) / fps

    fig = plt.figure(figsize=(14, 10))
    emotions = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]

    plt.subplot(4, 1, 1)
    for emo in emotions:
        values = [e[emo] for e in emotion_data]
        plt.plot(times, values, label=emo)
    plt.title("Emotion Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(times, face_features["frown"], label="Frown")
    plt.plot(times, face_features["jaw"], label="Jaw")
    plt.plot(times, face_features["mouth_open"], label="Mouth Open")
    plt.title("Facial Features")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(times, valence, label="Valence", color="blue")
    plt.title("Valence Over Time")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(times, face_features["comfort"], label="Comfort", color="green")
    plt.plot(times, face_features["saccade"], label="Saccade", color="purple")
    plt.title("Comfort & Saccade")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_over_time.png")
    plt.savefig(plot_path)
    plt.close()

    df = pd.DataFrame({
        "time": times,
        "valence": valence,
        "comfort": face_features["comfort"],
        "saccade": face_features["saccade"],
        "frown": face_features["frown"],
        "jaw": face_features["jaw"],
        "mouth_open": face_features["mouth_open"]
    })
    for emo in emotions:
        df[emo] = [e[emo] for e in emotion_data]

    csv_path = os.path.join(output_dir, "metrics_data.csv")
    df.to_csv(csv_path, index=False)
    return plot_path, csv_path