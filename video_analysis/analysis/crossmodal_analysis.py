from openai import OpenAI
import numpy as np
import os
import json
from scipy.signal import find_peaks

client = OpenAI()


def align_features(visual_features, audio_feat, text_feat, face_features, fps=30, target_fps=5):
    """
    Align all features to a common timeline with proper integer-based sampling
    """
    # Convert inputs to numpy arrays if they aren't already
    visual_features = np.array(visual_features, dtype=float)

    # For each face feature, ensure it's a numpy array
    for key in face_features:
        face_features[key] = np.array(face_features[key], dtype=float)

    # Get the minimum length of all input arrays to ensure consistency
    min_length = min(
        len(visual_features),
        len(face_features['saccade']),
        len(face_features['comfort']),
        len(face_features['frown']),
        len(face_features['jaw']),
        len(face_features['mouth_open'])
    )

    # Truncate all arrays to the minimum length
    visual_features = visual_features[:min_length]
    face_features['saccade'] = face_features['saccade'][:min_length]
    face_features['comfort'] = face_features['comfort'][:min_length]
    face_features['frown'] = face_features['frown'][:min_length]
    face_features['jaw'] = face_features['jaw'][:min_length]
    face_features['mouth_open'] = face_features['mouth_open'][:min_length]

    # Calculate the integer downsampling step
    fps = int(fps)  # Ensure fps is an integer
    target_fps = int(target_fps)  # Ensure target_fps is an integer
    step = max(1, fps // target_fps)

    # Calculate the output length after downsampling
    output_length = min_length // step

    # Create the indices for sampling
    indices = np.arange(0, output_length * step, step, dtype=int)

    # Ensure indices don't exceed array bounds
    if len(indices) > 0 and indices[-1] >= min_length:
        indices = indices[:-1]

    # Downsample all arrays using the same exact indices
    visual_aligned = visual_features[indices]
    saccade_aligned = face_features['saccade'][indices]
    frown_aligned = face_features['frown'][indices]
    jaw_aligned = face_features['jaw'][indices]
    mouth_open_aligned = face_features['mouth_open'][indices]
    raw_comfort_aligned = face_features['comfort'][indices]

    # Verify we have at least one point
    if len(indices) == 0:
        # If no valid indices, create at least one point
        indices = np.array([0])
        visual_aligned = np.array([visual_features[0]])
        saccade_aligned = np.array([face_features['saccade'][0]])
        frown_aligned = np.array([face_features['frown'][0]])
        jaw_aligned = np.array([face_features['jaw'][0]])
        mouth_open_aligned = np.array([face_features['mouth_open'][0]])
        raw_comfort_aligned = np.array([face_features['comfort'][0]])

    # Get the actual length of aligned arrays
    actual_length = len(visual_aligned)

    # For audio features (which are scalar), replicate across time
    # Normalize pitch to 0-1 range (assuming 80-300Hz normal range)
    normalized_pitch = np.clip((float(audio_feat[0]) - 80) / 220, 0, 1)
    pitch_aligned = np.full(actual_length, normalized_pitch)

    # Normalize intensity
    normalized_intensity = np.clip(float(audio_feat[1]) / 100, 0, 1)
    intensity_aligned = np.full(actual_length, normalized_intensity)

    # For text sentiment (scalar), replicate across time
    text_sentiment_aligned = np.full(actual_length, float(text_feat[0]))

    # Create a time array in seconds (using integer fps for consistency)
    time_array = indices.astype(float) / float(fps)

    # Create the aligned features dictionary
    aligned_features = {
        'time': time_array,
        'visual_valence': visual_aligned,
        'pitch': pitch_aligned,
        'intensity': intensity_aligned,
        'text_sentiment': text_sentiment_aligned,
        'eye_movement': saccade_aligned,
        'raw_comfort': raw_comfort_aligned,
        'frown': frown_aligned,
        'jaw': jaw_aligned,
        'mouth_open': mouth_open_aligned
    }

    # Final check to ensure all arrays have exactly the same length
    lengths = [len(aligned_features[key]) for key in aligned_features]
    if len(set(lengths)) != 1:
        min_len = min(lengths)
        for key in aligned_features:
            aligned_features[key] = aligned_features[key][:min_len]

    return aligned_features


def calculate_three_scores(features):
    """
    Calculate the three core emotional scores using weighted feature fusion
    """
    # Final check to ensure all arrays have exactly the same length
    arrays = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            arrays[key] = len(value)

    lengths = list(arrays.values())
    if len(set(lengths)) != 1:
        min_length = min(lengths)
        for key in arrays.keys():
            features[key] = features[key][:min_length]

    # 1. Emotion Valence Score - from facial emotions and text sentiment
    emotion_valence = (
        0.65 * features['visual_valence'] +
        0.35 * features['text_sentiment']
    )

    # 2. Comfort Level Score - from facial expressions
    # Method: Weighted fusion of facial movements and expressions
    # Weights: Eye movement (40%, inverted), Mouth openness (30%), Frown intensity (30%, inverted)
    comfort_level = (
        # Less eye movement = more comfort (40%)
        0.4 * (1.0 - features['eye_movement']) +
        # More natural mouth openness = more comfort (30%)
        0.3 * features['mouth_open'] +
        # Less frowning = more comfort (30%)
        0.3 * (1.0 - features['frown'])
    )

    # 3. Engagement Level Score - from eye movement and audio features
    # Method: Weighted fusion of eye movement and audio features
    # Weights: Eye movement (50%), Pitch (30%, normalized), Audio intensity (20%)
    engagement_level = (
        # More eye movement = more engagement (50%)
        0.5 * features['eye_movement'] +
        # Higher pitch often = more engagement (30%)
        0.3 * features['pitch'] +
        # Louder speech = more engagement (20%)
        0.2 * features['intensity']
    )

    # Calculate temporal dynamics for additional metrics
    valence_stability = 1.0 - min(1.0, np.std(emotion_valence) * 2)
    comfort_stability = 1.0 - min(1.0, np.std(comfort_level) * 2)
    engagement_stability = 1.0 - min(1.0, np.std(engagement_level) * 2)

    # Return all scores and metrics
    return {
        'emotion_valence': emotion_valence,
        'comfort_level': comfort_level,
        'engagement_level': engagement_level,
        'valence_stability': valence_stability,
        'comfort_stability': comfort_stability,
        'engagement_stability': engagement_stability
    }


def analyze_multimodal_features(text, text_feat, audio_feat, valence, saccade, comfort, face_features=None, fps=30):
    """
    Enhanced analysis with weighted feature fusion and time alignment
    """
    # Ensure fps is an integer (consistent with main.py)
    fps = int(fps)

    # Convert all inputs to numpy arrays
    valence = np.array(valence, dtype=float)
    saccade = np.array(saccade, dtype=float)
    comfort = np.array(comfort, dtype=float)

    # Print some debug info
    print(f"\nAnalyzing multimodal features:")
    print(f"  Valence array shape: {valence.shape}")
    print(f"  Saccade array shape: {saccade.shape}")
    print(f"  Comfort array shape: {comfort.shape}")
    print(f"  Text sentiment: {text_feat[0]:.2f}")
    print(
        f"  Audio features: pitch={audio_feat[0]:.2f}, intensity={audio_feat[1]:.2f}")

    # Ensure text_feat and audio_feat are numpy arrays
    if not isinstance(text_feat, np.ndarray):
        text_feat = np.array(text_feat, dtype=float)

    if not isinstance(audio_feat, np.ndarray):
        audio_feat = np.array(audio_feat, dtype=float)

    # Create face features dict if not provided
    if face_features is None:
        # Make sure all arrays have the same length by using the minimum length
        min_length = min(len(valence), len(saccade), len(comfort))
        valence = valence[:min_length]
        saccade = saccade[:min_length]
        comfort = comfort[:min_length]

        face_features = {
            'saccade': saccade,
            'comfort': comfort,
            'frown': np.zeros(min_length, dtype=float),
            'jaw': np.zeros(min_length, dtype=float),
            'mouth_open': np.zeros(min_length, dtype=float)
        }
    else:
        # Make sure all arrays have the same length
        array_lengths = {
            'valence': len(valence),
            'saccade': len(face_features.get('saccade', [])) if len(face_features.get('saccade', [])) > 0 else float('inf'),
            'comfort': len(face_features.get('comfort', [])) if len(face_features.get('comfort', [])) > 0 else float('inf'),
            'frown': len(face_features.get('frown', [])) if len(face_features.get('frown', [])) > 0 else float('inf'),
            'jaw': len(face_features.get('jaw', [])) if len(face_features.get('jaw', [])) > 0 else float('inf'),
            'mouth_open': len(face_features.get('mouth_open', [])) if len(face_features.get('mouth_open', [])) > 0 else float('inf')
        }

        print(f"  Face feature array lengths: {array_lengths}")

        valid_lengths = [
            length for length in array_lengths.values() if length != float('inf')]
        if not valid_lengths:
            min_length = len(valence)
        else:
            min_length = min(valid_lengths)

        print(f"  Using minimum length: {min_length}")

        valence = valence[:min_length]

        # Ensure all required keys exist in face_features
        for key in ['saccade', 'comfort', 'frown', 'jaw', 'mouth_open']:
            if key not in face_features or len(face_features[key]) == 0:
                face_features[key] = np.zeros(min_length, dtype=float)
            else:
                face_features[key] = face_features[key][:min_length]

    # 1. Align features to common 5fps timeline
    try:
        aligned_features = align_features(
            valence, audio_feat, text_feat, face_features, fps=fps
        )
        print(
            f"  Successfully aligned features to {len(aligned_features['time'])} time points")
    except Exception as e:
        print(f"Error in align_features: {e}")
        # Create a minimal aligned features dictionary with just a single point
        aligned_features = {
            'time': np.array([0.0]),
            'visual_valence': np.array([np.mean(valence)]),
            'pitch': np.array([audio_feat[0]]),
            'intensity': np.array([audio_feat[1]]),
            'text_sentiment': np.array([text_feat[0]]),
            'eye_movement': np.array([np.mean(saccade)]),
            'raw_comfort': np.array([np.mean(comfort)]),
            'frown': np.array([0.0]),
            'jaw': np.array([0.0]),
            'mouth_open': np.array([0.0])
        }
        print(f"  Created fallback aligned features with 1 time point")

    # 2. Calculate the three core scores
    try:
        scores = calculate_three_scores(aligned_features)
        print(f"  Successfully calculated three core scores")
    except Exception as e:
        print(f"Error in calculate_three_scores: {e}")
        # Create fallback scores with single values
        scores = {
            'emotion_valence': np.array([np.mean(valence)]),
            'comfort_level': np.array([np.mean(comfort)]),
            'engagement_level': np.array([np.mean(saccade)]),
            'valence_stability': 0.5,
            'comfort_stability': 0.5,
            'engagement_stability': 0.5
        }
        print(f"  Created fallback scores with 1 time point")

    # 3. Calculate averages for summary
    avg_valence = float(np.mean(scores['emotion_valence']))
    avg_comfort = float(np.mean(scores['comfort_level']))
    avg_engagement = float(np.mean(scores['engagement_level']))

    # Print some debug info on the scores
    print(
        f"  Average scores: valence={avg_valence:.2f}, comfort={avg_comfort:.2f}, engagement={avg_engagement:.2f}")
    print(f"  Score ranges: valence=[{np.min(scores['emotion_valence']):.2f}, {np.max(scores['emotion_valence']):.2f}], " +
          f"comfort=[{np.min(scores['comfort_level']):.2f}, {np.max(scores['comfort_level']):.2f}], " +
          f"engagement=[{np.min(scores['engagement_level']):.2f}, {np.max(scores['engagement_level']):.2f}]")

    # 4. Make sure text is a string, but don't truncate it
    transcript = ""
    if text is not None:
        if isinstance(text, str):
            transcript = text
        else:
            transcript = str(text)

    # Use GPT only for descriptive analysis, not for score calculation
    prompt = (
        f"Based on emotional analysis of a dating video, please provide a structured "
        f"feedback for the person with these emotional indicators:\n\n"
        f"Emotion Valence: {avg_valence:.2f} (stability: {scores['valence_stability']:.2f})\n"
        f"Comfort Level: {avg_comfort:.2f} (stability: {scores['comfort_stability']:.2f})\n"
        f"Engagement Level: {avg_engagement:.2f} (stability: {scores['engagement_stability']:.2f})\n\n"
        f"Text Transcript: {transcript}\n\n"
        f"Please provide actionable communication tips and insights about their "
        f"emotional expression. Focus on strengths and areas for improvement in dating "
        f"communication. Format your response with clear paragraphs and numbered points "
        f"when appropriate. Don't mention any technical terms like 'analysis', 'scores', or 'metrics'."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    analysis = response.choices[0].message.content

    # Safely convert all arrays to lists for JSON serialization
    try:
        valence_ts = scores['emotion_valence'].tolist()
        comfort_ts = scores['comfort_level'].tolist()
        engagement_ts = scores['engagement_level'].tolist()
        time_list = aligned_features['time'].tolist()
    except Exception as e:
        print(f"Error converting arrays to lists: {e}")
        # Fallback to single-point lists if conversion fails
        valence_ts = [float(avg_valence)]
        comfort_ts = [float(avg_comfort)]
        engagement_ts = [float(avg_engagement)]
        time_list = [0.0]

    # Create metrics dictionary with guaranteed serializable values
    metrics = {
        "text_sentiment": float(text_feat[0]),
        "pitch": float(audio_feat[0]),
        "intensity": float(audio_feat[1]),
        "valence_score": float(avg_valence),
        "comfort_score": float(avg_comfort),
        "engagement_score": float(avg_engagement),
        "valence_stability": float(scores['valence_stability']),
        "comfort_stability": float(scores['comfort_stability']),
        "engagement_stability": float(scores['engagement_stability']),

        # Time series data for real-time display
        # Explicitly convert each value to float to avoid numpy types
        "valence_ts": [float(v) for v in valence_ts],
        "comfort_ts": [float(c) for c in comfort_ts],
        "engagement_ts": [float(e) for e in engagement_ts],
        "time": [float(t) for t in time_list]
    }

    return analysis, metrics


def calculate_compatibility(video1_dir, video2_dir):
    """Calculate compatibility between two analyzed videos"""
    try:
        # Load analyses from both videos
        with open(os.path.join(video1_dir, "crossmodal_analysis.txt"), "r") as f:
            analysis1 = f.read()

        with open(os.path.join(video2_dir, "crossmodal_analysis.txt"), "r") as f:
            analysis2 = f.read()

        # Load metrics data
        with open(os.path.join(video1_dir, "metrics_data.json"), "r") as f:
            metrics1 = json.load(f)

        with open(os.path.join(video2_dir, "metrics_data.json"), "r") as f:
            metrics2 = json.load(f)

        # Calculate compatibility metrics

        # 1. Emotional synchrony (correlation between valence patterns)
        if 'valence_ts' in metrics1 and 'valence_ts' in metrics2 and len(metrics1['valence_ts']) > 0 and len(metrics2['valence_ts']) > 0:
            # Use the three core scores if available
            min_length = min(len(metrics1['valence_ts']), len(
                metrics2['valence_ts']))
            valence1 = np.array(metrics1['valence_ts'][:min_length])
            valence2 = np.array(metrics2['valence_ts'][:min_length])

            if min_length >= 2:  # Need at least 2 points for correlation
                valence_correlation = np.corrcoef(valence1, valence2)[0, 1]
            else:
                # Fallback if not enough points
                valence_correlation = 0.5  # Neutral correlation
        else:
            # Fall back to original valence if new scores aren't available
            min_length = min(
                len(metrics1['valence']), len(metrics2['valence']))
            if min_length >= 2:
                valence_correlation = np.corrcoef(
                    metrics1['valence'][:min_length],
                    metrics2['valence'][:min_length]
                )[0, 1]
            else:
                valence_correlation = 0.5

        # Handle NaN correlation values
        if np.isnan(valence_correlation):
            valence_correlation = 0.0

        # 2. Comfort synchrony (correlation between comfort levels)
        if 'comfort_ts' in metrics1 and 'comfort_ts' in metrics2 and len(metrics1['comfort_ts']) > 0 and len(metrics2['comfort_ts']) > 0:
            min_length = min(len(metrics1['comfort_ts']), len(
                metrics2['comfort_ts']))
            comfort1 = np.array(metrics1['comfort_ts'][:min_length])
            comfort2 = np.array(metrics2['comfort_ts'][:min_length])

            if min_length >= 2:  # Need at least 2 points for correlation
                comfort_correlation = np.corrcoef(comfort1, comfort2)[0, 1]
            else:
                comfort_correlation = 0.5
        else:
            min_length = min(
                len(metrics1['comfort']), len(metrics2['comfort']))
            if min_length >= 2:
                comfort_correlation = np.corrcoef(
                    metrics1['comfort'][:min_length],
                    metrics2['comfort'][:min_length]
                )[0, 1]
            else:
                comfort_correlation = 0.5

        # Handle NaN correlation values
        if np.isnan(comfort_correlation):
            comfort_correlation = 0.0

        # 3. Engagement balance (how evenly engaged both participants are)
        if 'engagement_score' in metrics1 and 'engagement_score' in metrics2:
            engagement1 = metrics1['engagement_score']
            engagement2 = metrics2['engagement_score']
        else:
            # Calculate from emotion data if not directly available
            engagement1 = 0.5  # Default value if unavailable
            engagement2 = 0.5  # Default value if unavailable

            if 'emotions' in metrics1 and 'emotions' in metrics2:
                # Safely calculate engagement from emotions
                try:
                    min_length = min(
                        len(metrics1['emotions']), len(metrics2['emotions']))

                    if min_length > 0:
                        engagement1 = np.mean([
                            e.get('happy', 0) + e.get('surprise', 0)
                            for e in metrics1['emotions'][:min_length]
                        ])
                        engagement2 = np.mean([
                            e.get('happy', 0) + e.get('surprise', 0)
                            for e in metrics2['emotions'][:min_length]
                        ])
                except Exception as e:
                    print(f"Error calculating engagement from emotions: {e}")

        engagement_balance = 1.0 - abs(float(engagement1) - float(engagement2))

        # 4. Emotional stability
        stability1 = metrics1.get('valence_stability', 0.5)
        stability2 = metrics2.get('valence_stability', 0.5)

        # 5. Mutual responsiveness (simplified calculation to avoid errors)
        responsiveness = 0.5  # Default middle value

        if 'valence_ts' in metrics1 and 'valence_ts' in metrics2 and len(metrics1['valence_ts']) > 3 and len(metrics2['valence_ts']) > 3:
            try:
                # Simplified responsiveness calculation that's less error-prone
                min_length = min(len(metrics1['valence_ts']), len(
                    metrics2['valence_ts']))
                valence1 = np.array(metrics1['valence_ts'][:min_length])
                valence2 = np.array(metrics2['valence_ts'][:min_length])

                # Count how often both time series move in the same direction
                # (both increasing or both decreasing)
                diff1 = np.diff(valence1)
                diff2 = np.diff(valence2)
                same_direction = np.sum((diff1 > 0) & (
                    diff2 > 0)) + np.sum((diff1 < 0) & (diff2 < 0))
                total_changes = len(diff1)

                if total_changes > 0:
                    responsiveness = float(
                        same_direction) / float(total_changes)
            except Exception as e:
                print(f"Error calculating responsiveness: {e}")

        # Use GPT for compatibility score calculation with improved formatting
        prompt = (
            f"Date 1 Emotional Profile:\n{analysis1}\n\n"
            f"Date 2 Emotional Profile:\n{analysis2}\n\n"
            f"Compatibility Indicators:\n"
            f"- Emotional Synchrony: {valence_correlation:.2f} (-1 to 1, higher is better)\n"
            f"- Comfort Synchrony: {comfort_correlation:.2f} (-1 to 1, higher is better)\n"
            f"- Engagement Balance: {engagement_balance:.2f} (0 to 1, higher means more balanced)\n"
            f"- Emotional Stability: Person 1 = {stability1:.2f}, Person 2 = {stability2:.2f} (0 to 1, higher is more stable)\n"
            f"- Mutual Responsiveness: {responsiveness:.2f} (0 to 1, higher means they respond to each other's emotions)\n\n"
            "Based on these emotional indicators for two people in a dating context, "
            "calculate a dating compatibility score from 0-100. Consider emotional "
            "synchrony, complementary traits, communication styles, and engagement levels. "
            "Provide just a single number score from 0-100, nothing else."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Extract score
        import re
        score_match = re.search(
            r'\b(\d{1,3})\b', response.choices[0].message.content)
        if score_match:
            score = int(score_match.group(1))
            if score > 100:  # Ensure score is within range
                score = 100
        else:
            # Default score if we can't extract one
            score = 75

        # Generate detailed compatibility analysis with improved formatting
        detailed_prompt = (
            f"Date 1 Emotional Profile:\n{analysis1}\n\n"
            f"Date 2 Emotional Profile:\n{analysis2}\n\n"
            f"Compatibility Indicators:\n"
            f"- Emotional Synchrony: {valence_correlation:.2f} (-1 to 1, higher is better)\n"
            f"- Comfort Synchrony: {comfort_correlation:.2f} (-1 to 1, higher is better)\n"
            f"- Engagement Balance: {engagement_balance:.2f} (0 to 1, higher means more balanced)\n"
            f"- Emotional Stability: Person 1 = {stability1:.2f}, Person 2 = {stability2:.2f} (0 to 1, higher is more stable)\n"
            f"- Mutual Responsiveness: {responsiveness:.2f} (0 to 1, higher means they respond to each other's emotions)\n\n"
            "Based on these emotional profiles, provide a detailed compatibility assessment "
            "explaining why these two people might or might not be compatible. Format your response "
            "with the following clearly labeled sections (exactly as shown, with '### ' prefix):\n\n"
            "### Emotional Patterns and Stability\n"
            "### Engagement Levels and Communication Style\n"
            "### Emotional Synchrony and Compatibility\n"
            "### Comfort and Mutual Responsiveness\n"
            "### Strengths and Recommendations\n"
            "### Conclusion\n\n"
            "Within each section, use bullet points starting with '- ' to highlight key points. "
            "Focus on communication aspects and provide concrete suggestions for improvement."
        )

        detailed_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": detailed_prompt}
            ],
            temperature=0.7
        )

        detailed_analysis = detailed_response.choices[0].message.content

        # Save both score and detailed analysis
        compatibility_data = {
            "score": score,
            "detailed_analysis": detailed_analysis,
            "metrics": {
                "emotional_synchrony": float(valence_correlation),
                "comfort_synchrony": float(comfort_correlation),
                "engagement_balance": float(engagement_balance),
                "emotional_stability_1": float(stability1),
                "emotional_stability_2": float(stability2),
                "mutual_responsiveness": float(responsiveness)
            }
        }

        with open(os.path.join(os.path.dirname(video1_dir), "compatibility_score.json"), "w") as f:
            json.dump(compatibility_data, f)

        return score, detailed_analysis

    except Exception as e:
        print(f"Error calculating compatibility: {e}")
        return 70, "Error generating detailed compatibility analysis."
