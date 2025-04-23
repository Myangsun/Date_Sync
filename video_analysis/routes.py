from flask import Blueprint, request, jsonify, current_app
import os
import json
from werkzeug.utils import secure_filename
import threading

# Import our video analysis function
from main import analyze_video, analyze_compatibility

api = Blueprint('api', __name__)


@api.route('/upload', methods=['POST'])
def upload_videos():
    """
    API endpoint to handle video uploads and start analysis process
    """
    # Check if the post request has the file part
    if 'video1' not in request.files or 'video2' not in request.files:
        return jsonify({'error': 'Missing video files'}), 400

    video1 = request.files['video1']
    video2 = request.files['video2']

    # Check if files are selected
    if video1.filename == '' or video2.filename == '':
        return jsonify({'error': 'No selected files'}), 400

    # Check if files are valid
    allowed_extensions = {'mp4', 'mov', 'avi', 'webm'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    if not allowed_file(video1.filename) or not allowed_file(video2.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    # Save files with secure filenames
    upload_folder = current_app.config['UPLOAD_FOLDER']
    video1_path = os.path.join(upload_folder, 'video_1.mp4')
    video2_path = os.path.join(upload_folder, 'video_2.mp4')

    video1.save(video1_path)
    video2.save(video2_path)

    # Start analysis in background thread
    thread = threading.Thread(
        target=process_videos,
        args=(video1_path, video2_path, current_app.config['OUTPUT_FOLDER'])
    )
    thread.start()

    return jsonify({
        'message': 'Videos uploaded successfully. Analysis started.',
        'status': 'processing'
    })


@api.route('/status', methods=['GET'])
def check_status():
    """
    Check the status of video processing
    """
    output_folder = current_app.config['OUTPUT_FOLDER']
    status_file = os.path.join(output_folder, 'status.json')

    if not os.path.exists(status_file):
        return jsonify({
            'status': 'not_started',
            'message': 'Analysis has not been started'
        })

    with open(status_file, 'r') as f:
        status_data = json.load(f)

    return jsonify(status_data)


@api.route('/results', methods=['GET'])
def get_results():
    """
    Get all analysis results after processing is complete
    """
    output_folder = current_app.config['OUTPUT_FOLDER']
    status_file = os.path.join(output_folder, 'status.json')

    if not os.path.exists(status_file):
        return jsonify({
            'status': 'not_started',
            'message': 'Analysis has not been started'
        }), 404

    with open(status_file, 'r') as f:
        status_data = json.load(f)

    if status_data['status'] != 'completed':
        return jsonify({
            'status': status_data['status'],
            'message': status_data['message'],
            'progress': status_data.get('progress', 0)
        })

    # Analysis completed, return all results
    try:
        # Load compatibility score
        with open(os.path.join(output_folder, 'compatibility_score.json'), 'r') as f:
            compatibility_data = json.load(f)

        # Load detailed analysis
        with open(os.path.join(output_folder, 'compatibility_analysis.txt'), 'r') as f:
            detailed_analysis = f.read()

        # Load metrics data for both videos
        with open(os.path.join(output_folder, 'video_1', 'metrics_data.json'), 'r') as f:
            metrics1 = json.load(f)

        with open(os.path.join(output_folder, 'video_2', 'metrics_data.json'), 'r') as f:
            metrics2 = json.load(f)

        # Load analysis text for both videos
        with open(os.path.join(output_folder, 'video_1', 'crossmodal_analysis.txt'), 'r') as f:
            analysis1 = f.read()

        with open(os.path.join(output_folder, 'video_2', 'crossmodal_analysis.txt'), 'r') as f:
            analysis2 = f.read()

        # Return all results
        return jsonify({
            'status': 'completed',
            'compatibility_score': compatibility_data['score'],
            'detailed_analysis': detailed_analysis,
            'video1': {
                'metrics': metrics1,
                'analysis': analysis1
            },
            'video2': {
                'metrics': metrics2,
                'analysis': analysis2
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving results: {str(e)}'
        }), 500


def process_videos(video1_path, video2_path, output_folder):
    """
    Process videos and update status file
    """
    status_file = os.path.join(output_folder, 'status.json')

    # Initialize status
    with open(status_file, 'w') as f:
        json.dump({
            'status': 'processing',
            'message': 'Analysis in progress',
            'progress': 0
        }, f)

    try:
        # Process first video
        update_status('processing', 'Analyzing first video', 25)
        video1_dir = analyze_video(video1_path, output_folder)

        # Process second video
        update_status('processing', 'Analyzing second video', 50)
        video2_dir = analyze_video(video2_path, output_folder)

        # Calculate compatibility
        update_status('processing', 'Calculating compatibility', 75)
        analyze_compatibility(video1_dir, video2_dir, output_folder)

        # Copy videos to output for display
        update_status('processing', 'Finalizing results', 90)
        os.system(f'cp {video1_path} {output_folder}/video_1.mp4')
        os.system(f'cp {video2_path} {output_folder}/video_2.mp4')

        # Update final status
        update_status('completed', 'Analysis completed', 100)

    except Exception as e:
        # Update error status
        with open(status_file, 'w') as f:
            json.dump({
                'status': 'error',
                'message': f'Error during analysis: {str(e)}'
            }, f)


def update_status(status, message, progress=0):
    """
    Update the status file with current progress
    """
    output_folder = current_app.config['OUTPUT_FOLDER']
    status_file = os.path.join(output_folder, 'status.json')

    with open(status_file, 'w') as f:
        json.dump({
            'status': status,
            'message': message,
            'progress': progress
        }, f)
