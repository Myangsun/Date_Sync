import numpy as np
from openai import OpenAI

client = OpenAI()

def extract_text_features(audio_path):
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text"
        )
    text = transcript.strip()

    prompt = f"Rate the sentiment of this text from -1 to 1:\n{text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    try:
        score = float(response.choices[0].message.content.strip())
    except:
        score = 0.0
    return text, np.array([score])