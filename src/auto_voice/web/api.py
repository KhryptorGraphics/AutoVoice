"""REST API endpoints for AutoVoice"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os

api_bp = Blueprint('api', __name__, url_prefix='/api')

UPLOAD_FOLDER = '/tmp/autovoice_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'AutoVoice API'})

@api_bp.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'path': filepath}), 200

    return jsonify({'error': 'Invalid file format'}), 400

@api_bp.route('/process', methods=['POST'])
def process_audio():
    data = request.json
    if not data or 'audio_path' not in data:
        return jsonify({'error': 'No audio path provided'}), 400

    # Placeholder for audio processing
    return jsonify({
        'status': 'processing',
        'audio_path': data['audio_path'],
        'message': 'Audio processing started'
    })

@api_bp.route('/synthesize', methods=['POST'])
def synthesize_voice():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Placeholder for voice synthesis
    return jsonify({
        'status': 'synthesizing',
        'text': data['text'],
        'message': 'Voice synthesis started'
    })