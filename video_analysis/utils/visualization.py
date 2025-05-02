import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI


def create_metrics_visualization(fps, valence, emotion_data, face_features, output_dir, core_scores=None):
    """
    Create visualization of emotional metrics over time
    Returns paths to PNG and CSV files
    """
    # Convert inputs to numpy arrays for safer handling
    valence = np.array(valence, dtype=float)

    # Check each face_features array has data and convert to numpy array
    for key in list(face_features.keys()):
        if len(face_features[key]) == 0:
            print(
                f"Warning: Empty face feature array '{key}'. Creating placeholder.")
            face_features[key] = np.zeros_like(valence)
        else:
            face_features[key] = np.array(face_features[key], dtype=float)

    # Find minimum length of all arrays to ensure consistency
    min_length = min(
        len(valence),
        len(face_features['comfort']),
        len(face_features['saccade']),
        len(face_features['frown']),
        len(face_features.get('jaw', [])
            ) if 'jaw' in face_features else float('inf'),
        len(face_features['mouth_open']),
        len(emotion_data)
    )

    if min_length == float('inf') or min_length == 0:
        print("Warning: Invalid array lengths detected. Using placeholder data.")
        min_length = 10
        valence = np.zeros(min_length)
        for key in face_features:
            face_features[key] = np.zeros(min_length)
        emotion_data = [{"happy": 0, "surprise": 0, "neutral": 1,
                         "sad": 0, "angry": 0, "fear": 0, "disgust": 0}] * min_length

    # Truncate all arrays to the minimum length
    valence = valence[:min_length]
    for key in face_features:
        face_features[key] = face_features[key][:min_length]
    emotion_data = emotion_data[:min_length]

    # Create time array with correct length
    time_array = np.arange(min_length) / fps

    # Create figure
    fig = plt.figure(figsize=(14, 12))  # Make taller for 4 plots

    # Plot 1: Valence and Valence Score
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(time_array, valence, 'r-', label='Raw Features')

    # Prepare core_scores data safely
    score_time = None
    if core_scores is not None:
        if "valence_ts" in core_scores and "time" in core_scores:
            if len(core_scores["valence_ts"]) > 0 and len(core_scores["time"]) > 0:
                # Ensure the core_scores arrays have matching lengths
                score_min_len = min(len(core_scores["time"]), len(
                    core_scores["valence_ts"]))
                score_time = np.array(core_scores["time"][:score_min_len])
                valence_ts = np.array(
                    core_scores["valence_ts"][:score_min_len])

                # Plot the new valence score
                ax1.plot(score_time, valence_ts, 'm-', linewidth=2,
                         label='Emotion Valence Score')

    ax1.set_ylabel('Value (-1 to 1)')
    ax1.set_title('Emotion Valence Score')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Comfort Level Score
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(time_array, face_features['comfort'], 'g-', label='Raw Features')

    # Plot the new comfort score if available
    if core_scores is not None and "comfort_ts" in core_scores and len(core_scores["comfort_ts"]) > 0 and score_time is not None:
        # Ensure matching length with score_time
        score_min_len = min(len(score_time), len(core_scores["comfort_ts"]))
        comfort_ts = np.array(core_scores["comfort_ts"][:score_min_len])
        ax2.plot(score_time[:score_min_len], comfort_ts,
                 'b-', linewidth=2, label='Comfort Level Score')

    ax2.set_ylabel('Value (-1 to 1)')
    ax2.set_title('Comfort Level Score')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Engagement Level Score
    ax3 = fig.add_subplot(4, 1, 3)

    # Plot the new engagement score if available
    if core_scores is not None and "engagement_ts" in core_scores and len(core_scores["engagement_ts"]) > 0 and score_time is not None:
        # Ensure matching length with score_time
        score_min_len = min(len(score_time), len(core_scores["engagement_ts"]))
        engagement_ts = np.array(core_scores["engagement_ts"][:score_min_len])
        ax3.plot(score_time[:score_min_len], engagement_ts,
                 'g-', linewidth=2, label='Engagement Level Score')
    else:
        # Fallback: use emotion data as proxy for engagement
        try:
            engagement = []
            for data in emotion_data:
                happy_val = data.get('happy', 0)
                surprise_val = data.get('surprise', 0)
                engagement.append(happy_val + surprise_val)

            engagement = np.array(engagement)
            ax3.plot(time_array, engagement, 'g-',
                     label='Estimated Engagement')
        except Exception as e:
            print(f"Warning: Could not create engagement plot: {e}")
            # Create placeholder data
            engagement = np.zeros(min_length)
            ax3.plot(time_array, engagement, 'g-',
                     label='Estimated Engagement (placeholder)')

    ax3.set_ylabel('Value (0 to 1)')
    ax3.set_title('Engagement Level Score')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Component Features
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(time_array, face_features['saccade'], 'b-', label='Eye Movement')
    ax4.plot(time_array, face_features['frown'], 'k-', label='Frown')
    ax4.plot(time_array, face_features['mouth_open'], 'c-', label='Mouth Open')

    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Value')
    ax4.set_title('Facial Feature Components')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'metrics_over_time.png')
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

    # Create and save CSV data safely
    try:
        # Create consistent length arrays for CSV data
        csv_data = {
            'time': time_array,
        }

        # Make sure all arrays have the same length
        for key in ['valence', 'comfort', 'saccade', 'frown', 'mouth_open']:
            if key == 'valence':
                csv_data[key] = valence
            else:
                csv_data[key] = face_features[key]

        # Add jaw if available
        if 'jaw' in face_features:
            csv_data['jaw'] = face_features['jaw'][:min_length]

        # Add emotion data with proper length checking
        emotions = ['happy', 'surprise', 'neutral',
                    'sad', 'angry', 'fear', 'disgust']
        for emotion in emotions:
            try:
                csv_data[emotion] = []
                for data in emotion_data:
                    csv_data[emotion].append(data.get(emotion, 0))
                # Ensure length matches
                csv_data[emotion] = csv_data[emotion][:min_length]
            except Exception as e:
                print(f"Warning: Error processing emotion '{emotion}': {e}")
                csv_data[emotion] = np.zeros(min_length)

        # Add core scores if available
        if core_scores is not None and score_time is not None:
            # Create separate dataframe for the downsampled data to avoid length issues
            score_data = {'time_5fps': score_time}
            if "valence_ts" in core_scores:
                score_min_len = min(len(score_time), len(
                    core_scores["valence_ts"]))
                score_data['valence_score'] = core_scores["valence_ts"][:score_min_len]
            if "comfort_ts" in core_scores:
                score_min_len = min(len(score_time), len(
                    core_scores["comfort_ts"]))
                score_data['comfort_score'] = core_scores["comfort_ts"][:score_min_len]
            if "engagement_ts" in core_scores:
                score_min_len = min(len(score_time), len(
                    core_scores["engagement_ts"]))
                score_data['engagement_score'] = core_scores["engagement_ts"][:score_min_len]

            # Save scores data to a separate CSV
            df_scores = pd.DataFrame(score_data)
            scores_csv_path = os.path.join(output_dir, 'scores_data.csv')
            df_scores.to_csv(scores_csv_path, index=False)

        # Create DataFrame for raw data
        df = pd.DataFrame(csv_data)

        # Save CSV
        csv_path = os.path.join(output_dir, 'metrics_data.csv')
        df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"Error creating CSV: {e}")
        # Create a simple fallback CSV with just time and valence
        csv_path = os.path.join(output_dir, 'metrics_data.csv')
        pd.DataFrame({'time': time_array, 'valence': valence}
                     ).to_csv(csv_path, index=False)

    return output_path, csv_path
