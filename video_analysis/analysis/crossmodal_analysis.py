from openai import OpenAI
import numpy as np
import os
import json

client = OpenAI()


def analyze_multimodal_features(text, text_feat, audio_feat, valence, saccade, comfort):
    """Analyze multimodal features from a single video and return analysis"""

    # Calculate average engagement (based on valence components)
    avg_valence = np.mean(valence)
    avg_saccade = np.mean(saccade)
    avg_comfort = np.mean(comfort)

    # Save these metrics for later use in compatibility calculation
    metrics = {
        "text_sentiment": float(text_feat[0]),
        "pitch": float(audio_feat[0]),
        "intensity": float(audio_feat[1]),
        "valence": float(avg_valence),
        "eye_movement": float(avg_saccade),
        "comfort": float(avg_comfort)
    }

    prompt = (
        f"Text Sentiment: {text_feat[0]:.2f}\n"
        f"Pitch: {audio_feat[0]:.1f} Hz, Intensity: {audio_feat[1]:.1f} dB\n"
        f"Visual Valence (avg): {avg_valence:.2f}\n"
        f"Eye Movement: {avg_saccade:.4f}\n"
        f"Comfort Level: {avg_comfort:.4f}\n"
        f"Transcript: {text[:300]}...\n\n"
        "Based on the above multimodal indicators, analyze the emotional state and suggest cross-cultural communication tips. "
        "Be specific about the person's comfort level, engagement level, and emotional status based on the metrics."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    analysis = response.choices[0].message.content

    return analysis


def calculate_compatibility(video1_dir, video2_dir):
    """Calculate compatibility between two analyzed videos"""

    # Load analyses from both videos
    try:
        with open(os.path.join(video1_dir, "crossmodal_analysis.txt"), "r") as f:
            analysis1 = f.read()

        with open(os.path.join(video2_dir, "crossmodal_analysis.txt"), "r") as f:
            analysis2 = f.read()

        prompt = (
            f"Video 1 Analysis:\n{analysis1}\n\n"
            f"Video 2 Analysis:\n{analysis2}\n\n"
            "Based on these multimodal emotional analyses of two people in a dating context, "
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

        # Extract just the score
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

        # Generate detailed compatibility analysis
        detailed_prompt = (
            f"Video 1 Analysis:\n{analysis1}\n\n"
            f"Video 2 Analysis:\n{analysis2}\n\n"
            "Based on these multimodal emotional analyses, provide a detailed compatibility assessment "
            "explaining why these two people might or might not be compatible. Consider emotional "
            "synchrony, complementary traits, communication styles, engagement levels, and comfort indicators. "
            "Include specific strengths and potential areas of growth for this potential relationship."
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
            "detailed_analysis": detailed_analysis
        }

        with open(os.path.join(os.path.dirname(video1_dir), "compatibility_analysis.json"), "w") as f:
            json.dump(compatibility_data, f)

        return score, detailed_analysis

    except Exception as e:
        print(f"Error calculating compatibility: {e}")
        return 70, "Error generating detailed compatibility analysis."
