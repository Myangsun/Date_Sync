import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_metrics_visualization(fps, valence, emotion_data, face_features, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if len(valence) == 0:
        print("⚠️ No valence data detected. Skipping visualization.")
        return None, None

    times = np.arange(len(valence)) / fps

    # 创建图
    fig = plt.figure(figsize=(14, 10))
    emotions = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]
    colors = ["green", "blue", "red", "purple", "brown", "orange", "gray"]

    # Plot 1: 每种情绪的时间曲线
    plt.subplot(4, 1, 1)
    for i, emotion in enumerate(emotions):
        emotion_values = [data[emotion] for data in emotion_data]
        plt.plot(times, emotion_values, label=emotion, color=colors[i])
    plt.title("Emotions Over Time")
    plt.ylabel("Probability")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # Plot 2: 面部细节特征
    plt.subplot(4, 1, 2)
    plt.plot(times, face_features["frown"], label="Frown")
    plt.plot(times, face_features["jaw"], label="Jaw")
    plt.plot(times, face_features["mouth_open"], label="Mouth Open")
    plt.title("Facial Muscle Features")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)

    # Plot 3: 情绪 Valence
    plt.subplot(4, 1, 3)
    plt.plot(times, valence, label="Valence", color="blue")
    plt.title("Emotion Valence Over Time")
    plt.grid(True, alpha=0.3)

    # Plot 4: 舒适度 + 眼动幅度
    plt.subplot(4, 1, 4)
    plt.plot(times, face_features["comfort"], label="Comfort", color="green")
    plt.plot(times, face_features["saccade"], label="Saccade", color="purple")
    plt.title("Comfort & Eye Movement")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_over_time.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # 保存 CSV
    df = pd.DataFrame({
        "time": times,
        "valence": valence,
        "saccade": face_features["saccade"],
        "comfort": face_features["comfort"],
        "frown": face_features["frown"],
        "jaw": face_features["jaw"],
        "mouth_open": face_features["mouth_open"],
    })

    for emo in emotions:
        df[emo] = [data[emo] for data in emotion_data]

    csv_path = os.path.join(output_dir, "metrics_data.csv")
    df.to_csv(csv_path, index=False)

    print(f"📈 图像保存: {plot_path}")
    print(f"📊 CSV保存: {csv_path}")

    return plot_path, csv_path
