from openai import OpenAI
import numpy as np

client = OpenAI()

def analyze_multimodal_features(text, text_feat, audio_feat, valence, saccade, comfort):
    prompt = (
        f"Transcript: \"{text}\"\n\n"
        f"Multimodal emotion indicators:\n"
        f"- Text sentiment: {text_feat[0]:.2f}\n"
        f"- Pitch: {audio_feat[0]:.1f} Hz, Intensity: {audio_feat[1]:.1f} dB\n"
        f"- Visual valence (avg): {np.mean(valence):.2f}\n"
        f"- Eye movement: {np.mean(saccade):.4f}\n"
        f"- Comfort level: {np.mean(comfort):.4f}\n\n"
        "Based on these signals, provide a cross-cultural emotional insight and give a communication suggestion."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in cross-cultural communication and affective computing."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    analysis = response.choices[0].message.content
    print("\n🧠 GPT-4 跨模态总结：\n", analysis)
    return analysis

