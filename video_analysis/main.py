import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from utils.video_io import extract_audio_and_frames
from analysis.text_features import extract_text_features
from analysis.audio_features import extract_audio_features
from analysis.visual_features import extract_visual_time_series
from analysis.face_mesh import extract_face_mesh_time_series
from analysis.crossmodal_analysis import analyze_multimodal_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['DATA_FOLDER']   = "static/data"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    # 接收文件
    f = request.files["video"]
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], "video.mp4")
    f.save(save_path)

    # 1) 提取音频+帧
    audio_path = os.path.join(app.config['DATA_FOLDER'], "audio.wav")
    frames_dir = os.path.join(app.config['DATA_FOLDER'], "frames")
    os.makedirs(frames_dir, exist_ok=True)
    total_frames, fps, duration = extract_audio_and_frames(
        save_path, audio_path, frames_dir
    )

    # 2) 文本情绪
    text, text_feat = extract_text_features(audio_path)

    # 3) 声学特征
    audio_feat = extract_audio_features(audio_path)

    # 4) 表情 valence
    valence, emotion_data = extract_visual_time_series(frames_dir)

    # 5) 面部特征
    saccade, comfort, face_features = extract_face_mesh_time_series(frames_dir)

    # 6) GPT 跨模态分析
    analysis = analyze_multimodal_features(
        text, text_feat, audio_feat, valence, saccade, comfort
    )

    # 7) 构造 metrics.json
    metrics = []
    for i, t in enumerate(np.arange(len(valence)) / fps):
        metrics.append({
            "time":    float(t),
            "valence": float(valence[i]),
            "comfort": float(comfort[i]),
            "engagement": float(saccade[i])
        })
    with open(os.path.join(app.config['DATA_FOLDER'], "metrics.json"), "w") as mf:
        json.dump(metrics, mf)

    # 8) 保存建议文本
    with open(os.path.join(app.config['DATA_FOLDER'], "analysis.txt"), "w") as af:
        af.write(analysis)

    return jsonify({"status": "ok"})


@app.route("/metrics")
def get_metrics():
    return app.send_static_file("data/metrics.json")


@app.route("/analysis")
def get_analysis():
    return app.send_static_file("data/analysis.txt")


if __name__ == "__main__":
    app.run(debug=True)
