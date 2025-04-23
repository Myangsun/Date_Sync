import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI


def create_metrics_visualization(fps, valence, emotion_data, face_features, output_dir):
    """
    Create visualization of emotional metrics over time
    Returns paths to PNG and CSV files
    """
    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # Time array (in seconds)
    time_array = np.arange(len(valence)) / fps

    # Plot valence
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time_array, valence, 'r-', label='Valence')
    ax1.plot(time_array, face_features['comfort'], 'g-', label='Comfort')
    ax1.set_ylabel('Value (-1 to 1)')
    ax1.set_title('Emotional Valence & Comfort Over Time')
    ax1.legend()
    ax1.grid(True)

    # Plot emotions
    ax2 = fig.add_subplot(3, 1, 2)
    emotions = ['happy', 'surprise', 'neutral',
                'sad', 'angry', 'fear', 'disgust']
    for emotion in emotions:
        values = [data[emotion] for data in emotion_data]
        ax2.plot(time_array, values, label=emotion.capitalize())
    ax2.set_ylabel('Probability')
    ax2.set_title('Emotion Probabilities Over Time')
    ax2.legend()
    ax2.grid(True)

    # Plot face metrics
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time_array, face_features['saccade'], 'b-', label='Eye Movement')
    ax3.plot(time_array, face_features['frown'], 'k-', label='Frown')
    ax3.plot(time_array, face_features['jaw'], 'm-', label='Jaw Distance')
    ax3.plot(time_array, face_features['mouth_open'], 'c-', label='Mouth Open')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Value')
    ax3.set_title('Facial Feature Metrics Over Time')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'metrics_over_time.png')
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

    # Create and save CSV data
    csv_data = {
        'time': time_array,
        'valence': valence,
        'comfort': face_features['comfort'],
        'saccade': face_features['saccade'],
        'frown': face_features['frown'],
        'jaw': face_features['jaw'],
        'mouth_open': face_features['mouth_open'],
    }

    # Add emotion data
    for emotion in emotions:
        csv_data[emotion] = [data[emotion] for data in emotion_data]

    # Create DataFrame
    df = pd.DataFrame(csv_data)

    # Save CSV
    csv_path = os.path.join(output_dir, 'metrics_data.csv')
    df.to_csv(csv_path, index=False)

    return output_path, csv_path
