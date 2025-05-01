from flask import render_template
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading

# Import our video analysis functions
from main import analyze_video, analyze_compatibility
from analysis.crossmodal_analysis import calculate_compatibility

app = Flask(__name__, static_folder='static')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if request has files
    if 'video1' not in request.files or 'video2' not in request.files:
        return jsonify({'error': 'Both video files are required'}), 400

    video1 = request.files['video1']
    video2 = request.files['video2']

    # Check if filenames are empty
    if video1.filename == '' or video2.filename == '':
        return jsonify({'error': 'No selected files'}), 400

    # Check if files are allowed
    if not (allowed_file(video1.filename) and allowed_file(video2.filename)):
        return jsonify({'error': 'Invalid file type, only videos are allowed'}), 400

    # Secure the filenames
    video1_filename = secure_filename(video1.filename)
    video2_filename = secure_filename(video2.filename)

    # Save the files
    video1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video_1.mp4')
    video2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video_2.mp4')

    video1.save(video1_path)
    video2.save(video2_path)

    # Start analysis in a separate thread to avoid timeout
    thread = threading.Thread(target=process_videos_and_calculate_compatibility,
                              args=(video1_path, video2_path))
    thread.start()

    # Return success response
    return jsonify({
        'message': 'Videos uploaded successfully, analysis started',
        'video1': video1_filename,
        'video2': video2_filename
    })


def process_videos_and_calculate_compatibility(video1_path, video2_path):
    """Process both videos and calculate compatibility score"""
    try:
        # Create status file
        update_status('processing', 'Starting analysis', 0)

        # Process both videos
        update_status('processing', 'Analyzing first video', 25)
        output_dir1 = analyze_video(video1_path, app.config['OUTPUT_FOLDER'])

        update_status('processing', 'Analyzing second video', 50)
        output_dir2 = analyze_video(video2_path, app.config['OUTPUT_FOLDER'])

        # Calculate compatibility score
        update_status('processing', 'Calculating compatibility', 75)
        score, detailed_analysis = analyze_compatibility(
            output_dir1, output_dir2, app.config['OUTPUT_FOLDER'])

        # Copy videos to output folders for display
        update_status('processing', 'Finalizing results', 90)
        os.system(
            f'cp {video1_path} {app.config["OUTPUT_FOLDER"]}/video_1.mp4')
        os.system(
            f'cp {video2_path} {app.config["OUTPUT_FOLDER"]}/video_2.mp4')

        update_status('completed', 'Analysis completed', 100)
        print(f"Analysis complete! Compatibility score: {score}")

    except Exception as e:
        print(f"Error in analysis: {e}")
        update_status('error', f'Error during analysis: {str(e)}', 0)


def update_status(status, message, progress=0):
    """Update the status file with current progress"""
    status_file = os.path.join(app.config['OUTPUT_FOLDER'], 'status.json')
    with open(status_file, 'w') as f:
        json.dump({
            'status': status,
            'message': message,
            'progress': progress
        }, f)


@app.route('/status', methods=['GET'])
def check_status():
    """Check the status of video processing"""
    status_file = os.path.join(app.config['OUTPUT_FOLDER'], 'status.json')

    if not os.path.exists(status_file):
        return jsonify({
            'status': 'not_started',
            'message': 'Analysis has not been started'
        })

    with open(status_file, 'r') as f:
        status_data = json.load(f)

    return jsonify(status_data)


@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    print("Starting Flask server on http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)
