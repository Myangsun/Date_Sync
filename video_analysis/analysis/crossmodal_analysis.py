from openai import OpenAI
import numpy as np

client = OpenAI()

def analyze_multimodal_features(text, text_feat, audio_feat, valence, saccade, comfort):
    prompt = (
        f"Text Sentiment: {text_feat[0]:.2f}\n"
        f"Pitch: {audio_feat[0]:.1f} Hz, Intensity: {audio_feat[1]:.1f} dB\n"
        f"Visual Valence (avg): {np.mean(valence):.2f}\n"
        f"Eye Movement: {np.mean(saccade):.4f}\n"
        f"Comfort Level: {np.mean(comfort):.4f}\n"
        f"Transcript: {text[:300]}...\n\n"
        "Based on the above multimodal indicators, analyze the emotional state and suggest cross-cultural communication tips."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content