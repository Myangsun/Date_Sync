from flask import render_template
import os
import json
from flask import Flask, request, jsonify, send_from_directory
import threading
from flask_socketio import SocketIO, join_room, leave_room, emit
import datetime
from werkzeug.utils import secure_filename

# Import our video analysis functions
from main import analyze_video, analyze_compatibility
from media_handler import process_recorded_videos

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload size

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}

# Socket.io events for WebRTC signaling


@socketio.on('connect')
def handle_connect():
    print('Client connected', request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected', request.sid)


@socketio.on('create-room')
def handle_create_room(data):
    room_id = data['roomId']
    join_room(room_id)
    print(f'Room created: {room_id}')
    emit('room-created', {'roomId': room_id})


@socketio.on('join-room')
def handle_join_room(data):
    room_id = data['roomId']
    join_room(room_id)
    print(f'User joined room: {room_id}')
    emit('room-joined', {'roomId': room_id})


@socketio.on('offer')
def handle_offer(data):
    room_id = data['roomId']
    print(f'Sending offer to room: {room_id}')
    emit('offer', {'offer': data['offer']}, room=room_id, skip_sid=request.sid)


@socketio.on('answer')
def handle_answer(data):
    room_id = data['roomId']
    print(f'Sending answer to room: {room_id}')
    emit('answer', {'answer': data['answer']},
         room=room_id, skip_sid=request.sid)


@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    room_id = data['roomId']
    print(f'Sending ICE candidate to room: {room_id}')
    emit('new-ice-candidate',
         {'candidate': data['candidate']}, room=room_id, skip_sid=request.sid)


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

    # Get file extensions
    ext1 = os.path.splitext(video1.filename)[1].lower()
    ext2 = os.path.splitext(video2.filename)[1].lower()

    upload_folder = app.config['UPLOAD_FOLDER']
    video1_path = os.path.join(upload_folder, 'video_1.mp4')
    video2_path = os.path.join(upload_folder, 'video_2.mp4')

    # Check if videos are recorded (webm format)
    if ext1 == '.webm' or ext2 == '.webm':
        # Process recorded videos
        video1_path, video2_path = process_recorded_videos(
            video1, video2, upload_folder
        )
    else:
        # Handle traditional uploads - check if files are valid
        if not (allowed_file(video1.filename) and allowed_file(video2.filename)):
            return jsonify({'error': 'Invalid file type, only videos are allowed'}), 400

        # Save the files
        video1.save(video1_path)
        video2.save(video2_path)

    # Start analysis in a separate thread to avoid timeout
    base_output_folder = app.config['OUTPUT_FOLDER']
    thread = threading.Thread(target=process_videos_and_calculate_compatibility,
                              args=(video1_path, video2_path, base_output_folder))
    thread.start()

    # Return success response
    return jsonify({
        'message': 'Videos uploaded successfully, analysis started',
        'video1': video1.filename,
        'video2': video2.filename
    })

# Function to create a timestamped output folder


def create_timestamped_output_folder(base_folder):
    """
    Creates a new output folder with timestamp

    Args:
        base_folder: Base output directory

    Returns:
        tuple: (Path to the new output folder, timestamp string)
    """
    # Create timestamp string: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create folder name, e.g., "output_2023-11-22_14-35-22"
    folder_name = f"output_{timestamp}"

    # Create full path
    output_folder = os.path.join(base_folder, folder_name)

    # Create the directory
    os.makedirs(output_folder, exist_ok=True)

    # Create subdirectories for each video
    os.makedirs(os.path.join(output_folder, "video_1"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "video_2"), exist_ok=True)

    print(f"Created output folder: {output_folder}")
    return output_folder, timestamp

# Modified process_videos_and_calculate_compatibility function


def process_videos_and_calculate_compatibility(video1_path, video2_path, base_output_folder):
    """Process both videos and calculate compatibility score"""
    try:
        # Create a new output folder with timestamp
        output_folder, timestamp = create_timestamped_output_folder(
            base_output_folder)

        # Create status file
        status_file = os.path.join(output_folder, 'status.json')
        with open(status_file, 'w') as f:
            json.dump({
                'status': 'processing',
                'message': 'Analysis in progress',
                'progress': 0,
                'output_folder': output_folder  # Store the output folder path in status
            }, f)

        # Process both videos
        update_status(output_folder, 'processing', 'Analyzing first video', 25)
        video1_dir = analyze_video(video1_path, output_folder)

        update_status(output_folder, 'processing',
                      'Analyzing second video', 50)
        video2_dir = analyze_video(video2_path, output_folder)

        # Calculate compatibility
        update_status(output_folder, 'processing',
                      'Calculating compatibility', 75)
        score, detailed_analysis = analyze_compatibility(
            video1_dir, video2_dir, output_folder)

        # Copy videos to output folders for display
        update_status(output_folder, 'processing', 'Finalizing results', 90)
        os.system(f'cp {video1_path} {output_folder}/video_1.mp4')
        os.system(f'cp {video2_path} {output_folder}/video_2.mp4')

        update_status(output_folder, 'completed', 'Analysis completed', 100)
        print(f"Analysis complete! Compatibility score: {score}")

        # Update latest.json to point to the most recent analysis
        latest_file = os.path.join(base_output_folder, 'latest.json')
        with open(latest_file, 'w') as f:
            json.dump({
                'output_folder': output_folder,
                'timestamp': timestamp
            }, f)

    except Exception as e:
        print(f"Error in analysis: {e}")
        if 'output_folder' in locals():
            update_status(output_folder, 'error',
                          f'Error during analysis: {str(e)}', 0)
        else:
            print("Error occurred before output folder was created")


def update_status(output_folder, status, message, progress=0):
    """Update the status file with current progress"""
    status_file = os.path.join(output_folder, 'status.json')
    with open(status_file, 'w') as f:
        json.dump({
            'status': status,
            'message': message,
            'progress': progress,
            'output_folder': output_folder
        }, f)


@app.route('/status', methods=['GET'])
def check_status():
    """Check the status of video processing"""
    base_output_folder = app.config['OUTPUT_FOLDER']

    # First, try to get the latest output folder
    latest_file = os.path.join(base_output_folder, 'latest.json')

    if os.path.exists(latest_file):
        try:
            with open(latest_file, 'r') as f:
                latest_data = json.load(f)
                output_folder = latest_data['output_folder']
                status_file = os.path.join(output_folder, 'status.json')

                if os.path.exists(status_file):
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                    return jsonify(status_data)
        except Exception as e:
            print(f"Error checking latest status: {e}")

    # If latest.json doesn't exist or has an error, check if there's any processing folder
    try:
        folders = [f for f in os.listdir(
            base_output_folder) if f.startswith('output_')]
        folders.sort(reverse=True)  # Most recent first

        for folder in folders:
            potential_folder = os.path.join(base_output_folder, folder)
            status_file = os.path.join(potential_folder, 'status.json')

            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)

                    if status_data['status'] != 'completed':
                        return jsonify(status_data)
                except Exception:
                    pass
    except Exception as e:
        print(f"Error searching for status files: {e}")

    # If no processing is found, return not started
    return jsonify({
        'status': 'not_started',
        'message': 'Analysis has not been started'
    })

# Updated output route to handle timestamped folders


@app.route('/output/<path:filename>')
def download_file(filename):
    base_output_folder = app.config['OUTPUT_FOLDER']

    # Check if it's a special path with folder prefix
    if '/' in filename:
        folder, file = filename.split('/', 1)
        if folder.startswith('output_'):
            folder_path = os.path.join(base_output_folder, folder)
            if os.path.exists(folder_path):
                return send_from_directory(folder_path, file)

    # Check latest output folder
    latest_file = os.path.join(base_output_folder, 'latest.json')
    if os.path.exists(latest_file):
        try:
            with open(latest_file, 'r') as f:
                latest_data = json.load(f)
                output_folder = latest_data['output_folder']
                if os.path.exists(os.path.join(output_folder, filename)):
                    return send_from_directory(output_folder, filename)
        except Exception as e:
            print(f"Error accessing latest output: {e}")

    # Fallback to the base output folder
    return send_from_directory(base_output_folder, filename)


if __name__ == '__main__':
    print("Starting Flask server with SocketIO on http://localhost:8080")
    socketio.run(app, debug=True, host='0.0.0.0',
                 port=8080, allow_unsafe_werkzeug=True)
