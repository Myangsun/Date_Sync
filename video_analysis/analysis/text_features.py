# analysis/text_features.py

import numpy as np
from openai import OpenAI

client = OpenAI()

def extract_text_features(audio_file):
    """Uses Whisper + GPT to get a single sentiment score [-1,1]."""

    # Step 1: Transcribe audio to text using Whisper
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text"
        )
    text = transcription.strip()
    print("Transcript preview:", text[:100], "...")

    # Step 2: Ask GPT to evaluate sentiment score [-1, 1]
    prompt = (
        "Provide a single sentiment score between -1 (very negative) and +1 (very positive), "
        "for the following transcript. **Return only that number.**\n\n"
        f"Transcript: {text}"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment-scoring tool."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        score = float(resp.choices[0].message.content.strip())
    except Exception:
        print("⚠️ Could not parse, defaulting to 0.")
        score = 0.0

    text_feat = np.array([score])
    print("Text feature:", text_feat)

    return text, text_feat
