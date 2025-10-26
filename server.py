"""
USPSA Shot Detection Server
Flask API for ML-based gunshot detection in videos
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for local testing

# Configuration
# Try .keras first, fallback to .h5 for compatibility
import os as _os
if _os.path.exists('shot_detector_model.keras'):
    MODEL_PATH = 'shot_detector_model.keras'
elif _os.path.exists('shot_detector_model.h5'):
    MODEL_PATH = 'shot_detector_model.h5'
else:
    MODEL_PATH = 'shot_detector_model.keras'  # Default

NORMALIZATION_PARAMS_PATH = 'normalization_params.json'
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm', 'mkv'}

# Audio processing parameters (must match training)
SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.15
N_MELS = 64
SPEC_SIZE = 64

# Model thresholds for 3-class detection
SHOT_THRESHOLD = 0.5  # Threshold for shot class probability
BEEP_THRESHOLD = 0.5  # Threshold for beep class probability

# Load normalization parameters
print("Loading normalization parameters...")
import json

try:
    with open(NORMALIZATION_PARAMS_PATH, 'r') as f:
        norm_params = json.load(f)
    GLOBAL_MIN = norm_params['global_min']
    GLOBAL_MAX = norm_params['global_max']
    print(f"✅ Normalization parameters loaded:")
    print(f"   Global min: {GLOBAL_MIN:.4f}")
    print(f"   Global max: {GLOBAL_MAX:.4f}")
except Exception as e:
    print(f"❌ Failed to load normalization parameters: {e}")
    print(f"   Using fallback per-segment normalization (may reduce accuracy)")
    GLOBAL_MIN = None
    GLOBAL_MAX = None

# Load the trained model
print("Loading ML model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print(f"   Make sure {MODEL_PATH} exists and was trained with compatible Keras version")
    model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio_segment(video_path, timestamp, duration=SEGMENT_DURATION, sr=SAMPLE_RATE):
    """
    Extract audio segment from video at specified timestamp
    Returns mel spectrogram matching training format

    **CRITICAL**: Uses fixed dB reference (ref=1.0) and global normalization
    to preserve amplitude differences between loud shots and quiet background
    """
    try:
        # Load audio from video
        y, _ = librosa.load(video_path, sr=sr, offset=timestamp, duration=duration, mono=True)

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            fmax=8000,
            hop_length=512
        )

        # **CRITICAL FIX**: Use FIXED reference value (ref=1.0) instead of np.max
        # This preserves absolute amplitude differences
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)

        # Resize to fixed dimensions
        if mel_spec_db.shape[1] < SPEC_SIZE:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, SPEC_SIZE - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :SPEC_SIZE]

        # **CRITICAL FIX**: Use GLOBAL normalization with training parameters
        if GLOBAL_MIN is not None and GLOBAL_MAX is not None:
            # Use global normalization parameters from training
            mel_spec_db = (mel_spec_db - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + 1e-8)
        else:
            # Fallback to per-segment normalization (less accurate)
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        return mel_spec_db

    except Exception as e:
        print(f"Error extracting audio at {timestamp}s: {e}")
        return None


def detect_beep(video_path, first_shot_time):
    """
    Detect timer beep using the original algorithm
    Scans first 10 seconds for sustained loud peak
    """
    try:
        # Load audio from video
        y, sr = librosa.load(video_path, sr=SAMPLE_RATE, duration=min(10, first_shot_time), mono=True)

        if len(y) == 0:
            return 0.0

        # Find global max
        global_max = np.max(np.abs(y))
        if global_max == 0:
            return 0.0

        # Scan for sustained peaks (beep characteristics)
        window_size = int(sr * 0.005)
        scan_step = int(sr * 0.05)
        search_end = min(len(y) - window_size, int((first_shot_time - 0.8) * sr))

        peaks = []
        for i in range(0, search_end, scan_step):
            max_val = np.max(np.abs(y[i:i + window_size]))
            amp = max_val / global_max

            if amp >= 0.05:
                time_s = i / sr
                peaks.append({'time': time_s, 'amp': amp})

        # Group consecutive peaks into events
        events = []
        current_event = None

        for peak in peaks:
            if current_event is None or peak['time'] - current_event['end_time'] > 0.1:
                if current_event:
                    events.append(current_event)
                current_event = {
                    'start_time': peak['time'],
                    'end_time': peak['time'],
                    'max_amp': peak['amp']
                }
            else:
                current_event['end_time'] = peak['time']
                current_event['max_amp'] = max(current_event['max_amp'], peak['amp'])

        if current_event:
            events.append(current_event)

        # Find first sustained event (beep)
        for event in events:
            duration = event['end_time'] - event['start_time']
            if duration >= 0.1 and duration <= 0.6 and event['max_amp'] >= 0.08 and event['max_amp'] <= 0.50:
                return event['start_time']

        return 0.0

    except Exception as e:
        print(f"Error detecting beep: {e}")
        return 0.0


def scan_video_for_shots_and_beeps(video_path, scan_interval=0.05, shot_threshold=None, beep_threshold=None):
    """
    Scan entire video using 3-class ML model to detect shots and beeps

    Args:
        video_path: Path to video file
        scan_interval: Time between scan points (seconds)
        shot_threshold: Detection threshold for shots (uses SHOT_THRESHOLD if None)
        beep_threshold: Detection threshold for beeps (uses BEEP_THRESHOLD if None)

    Returns:
        Tuple of (shot_detections, beep_detections)
    """
    if shot_threshold is None:
        shot_threshold = SHOT_THRESHOLD
    if beep_threshold is None:
        beep_threshold = BEEP_THRESHOLD

    try:
        # Get video duration
        duration = librosa.get_duration(path=video_path)

        shot_detections = []
        beep_detections = []
        scan_times = np.arange(0, duration - SEGMENT_DURATION, scan_interval)

        print(f"Scanning {len(scan_times)} positions in {duration:.1f}s video...", flush=True)
        print(f"Using shot threshold: {shot_threshold}, beep threshold: {beep_threshold}", flush=True)

        max_shot_prob = 0.0
        max_beep_prob = 0.0
        shot_predictions_above_01 = 0
        beep_predictions_above_01 = 0

        for i, timestamp in enumerate(scan_times):
            # Progress indicator
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(scan_times)} ({i/len(scan_times)*100:.1f}%)", flush=True)

            # Extract audio features
            spec = extract_audio_segment(video_path, timestamp)
            if spec is None:
                continue

            # Prepare for model
            spec_input = spec.reshape(1, N_MELS, SPEC_SIZE, 1)

            # Get 3-class prediction: [prob_nonshot, prob_shot, prob_beep]
            predictions = model.predict(spec_input, verbose=0)[0]
            prob_nonshot = predictions[0]
            prob_shot = predictions[1]
            prob_beep = predictions[2]

            # Track statistics
            max_shot_prob = max(max_shot_prob, prob_shot)
            max_beep_prob = max(max_beep_prob, prob_beep)

            if prob_shot > 0.1:
                shot_predictions_above_01 += 1
            if prob_beep > 0.1:
                beep_predictions_above_01 += 1

            # If confident it's a shot, record it
            if prob_shot > shot_threshold:
                shot_detections.append({
                    'time': float(timestamp),
                    'confidence': float(prob_shot)
                })

            # If confident it's a beep, record it
            if prob_beep > beep_threshold:
                beep_detections.append({
                    'time': float(timestamp),
                    'confidence': float(prob_beep)
                })

        print(f"  Max shot probability: {max_shot_prob:.4f}", flush=True)
        print(f"  Max beep probability: {max_beep_prob:.4f}", flush=True)
        print(f"  Shot predictions > 0.1: {shot_predictions_above_01}", flush=True)
        print(f"  Beep predictions > 0.1: {beep_predictions_above_01}", flush=True)

        print(f"✅ Found {len(shot_detections)} potential shots, {len(beep_detections)} potential beeps", flush=True)

        # Merge nearby detections
        merged_shots = merge_nearby_detections(shot_detections, merge_window=0.1)
        merged_beeps = merge_nearby_detections(beep_detections, merge_window=0.2)

        print(f"✅ After merging: {len(merged_shots)} shots, {len(merged_beeps)} beeps", flush=True)

        return merged_shots, merged_beeps

    except Exception as e:
        print(f"Error scanning video: {e}")
        return [], []


def merge_nearby_detections(detections, merge_window=0.2):
    """
    Merge detections that are within merge_window seconds of each other
    Keeps the one with highest confidence
    """
    if not detections:
        return []

    # Sort by time
    detections = sorted(detections, key=lambda x: x['time'])

    merged = []
    current_group = [detections[0]]

    for det in detections[1:]:
        if det['time'] - current_group[-1]['time'] < merge_window:
            # Add to current group
            current_group.append(det)
        else:
            # Start new group, save previous
            # Keep detection with highest confidence
            best = max(current_group, key=lambda x: x['confidence'])
            merged.append(best)
            current_group = [det]

    # Don't forget last group
    if current_group:
        best = max(current_group, key=lambda x: x['confidence'])
        merged.append(best)

    return merged


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Main endpoint: Analyze uploaded video for shots

    Returns JSON with:
        - beep_time: Detected timer beep timestamp
        - shots: List of detected shot timestamps with confidence
        - total_shots: Total number of shots detected
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Check if file was uploaded
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, mov, avi, webm, mkv'}), 400

    try:
        # Get threshold from request (optional)
        threshold = request.form.get('threshold', type=float)
        if threshold is None:
            threshold = SHOT_THRESHOLD

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        print(f"\n{'='*50}", flush=True)
        print(f"Processing: {filename}", flush=True)
        print(f"Shot threshold: {threshold}", flush=True)
        print(f"{'='*50}", flush=True)

        # Step 1: Scan for shots AND beeps using 3-class ML model
        print("\n🔍 Step 1: Scanning for shots and beeps using ML...", flush=True)
        shots, beeps = scan_video_for_shots_and_beeps(temp_path, scan_interval=0.05, shot_threshold=threshold)

        # Step 2: Get beep time from ML detections
        print("\n🔔 Step 2: Processing beep detections...", flush=True)
        if beeps:
            # Use the first detected beep
            beep_time = beeps[0]['time']
            print(f"  ML-detected beep at: {beep_time:.2f}s (confidence: {beeps[0]['confidence']:.2f})", flush=True)
        else:
            # Fallback to signal processing if ML didn't detect beep
            print(f"  No ML beep detected, trying signal processing fallback...", flush=True)
            if shots:
                first_shot_time = shots[0]['time']
                beep_time = detect_beep(temp_path, first_shot_time)
                print(f"  Signal processing beep at: {beep_time:.2f}s", flush=True)
            else:
                beep_time = 0.0
                print(f"  No beep detected", flush=True)

        # Step 3: Filter shots after beep
        shots_after_beep = [s for s in shots if s['time'] > beep_time + 0.05]

        # Calculate draw time
        draw_time = shots_after_beep[0]['time'] - beep_time if shots_after_beep else 0.0

        print(f"\n📊 Results:", flush=True)
        print(f"  Beep: {beep_time:.2f}s", flush=True)
        print(f"  Shots detected: {len(shots_after_beep)}", flush=True)
        print(f"  Draw time: {draw_time:.2f}s", flush=True)
        print(f"{'='*50}\n", flush=True)

        # Clean up temp file
        os.remove(temp_path)

        # Return results
        return jsonify({
            'success': True,
            'beep_time': beep_time,
            'shots': shots_after_beep,
            'total_shots': len(shots_after_beep),
            'draw_time': draw_time
        })

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert-to-mp4', methods=['POST'])
def convert_to_mp4():
    """
    Convert WebM video to MP4 using FFmpeg.
    Accepts a WebM file upload and returns job ID for polling.
    """
    import subprocess
    import uuid
    import threading

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Create unique job ID
        job_id = str(uuid.uuid4())

        # Save WebM file
        webm_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_input.webm')
        mp4_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_output.mp4')
        video_file.save(webm_path)

        # Initialize job status
        conversion_jobs[job_id] = {
            'progress': 0,
            'complete': False,
            'error': None,
            'mp4_path': mp4_path
        }

        # Start conversion in background thread
        def convert():
            try:
                # Find ffprobe and ffmpeg executables
                import shutil
                ffprobe_path = shutil.which('ffprobe') or '/usr/local/bin/ffprobe' or '/opt/homebrew/bin/ffprobe'
                ffmpeg_path = shutil.which('ffmpeg') or '/usr/local/bin/ffmpeg' or '/opt/homebrew/bin/ffmpeg'

                if not ffprobe_path or not os.path.exists(ffprobe_path):
                    raise FileNotFoundError('ffprobe not found. Please install FFmpeg: brew install ffmpeg')
                if not ffmpeg_path or not os.path.exists(ffmpeg_path):
                    raise FileNotFoundError('ffmpeg not found. Please install FFmpeg: brew install ffmpeg')

                # Get video duration for progress tracking
                duration_cmd = [
                    ffprobe_path, '-v', 'error', '-show_entries',
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    webm_path
                ]
                duration = float(subprocess.check_output(duration_cmd).decode().strip())

                # Convert WebM to MP4 with FFmpeg
                cmd = [
                    ffmpeg_path, '-i', webm_path,
                    '-c:v', 'libx264',  # H.264 video codec
                    '-preset', 'medium',  # Balance between speed and quality
                    '-crf', '23',  # Quality (lower = better, 18-28 is good range)
                    '-c:a', 'aac',  # AAC audio codec
                    '-b:a', '192k',  # Audio bitrate
                    '-movflags', '+faststart',  # Enable streaming
                    '-progress', 'pipe:1',  # Output progress to stdout
                    '-y',  # Overwrite output file
                    mp4_path
                ]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )

                # Parse progress
                for line in process.stdout:
                    if line.startswith('out_time_ms='):
                        time_ms = int(line.split('=')[1])
                        progress = min(99, (time_ms / 1000000 / duration) * 100)
                        conversion_jobs[job_id]['progress'] = progress

                process.wait()

                if process.returncode == 0:
                    conversion_jobs[job_id]['progress'] = 100
                    conversion_jobs[job_id]['complete'] = True
                    # Clean up WebM
                    os.remove(webm_path)
                else:
                    error_msg = process.stderr.read()
                    conversion_jobs[job_id]['error'] = f'FFmpeg error: {error_msg}'

            except Exception as e:
                conversion_jobs[job_id]['error'] = str(e)

        thread = threading.Thread(target=convert)
        thread.start()

        return jsonify({
            'success': True,
            'jobId': job_id,
            'statusUrl': f'/api/conversion-status/{job_id}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversion-status/<job_id>', methods=['GET'])
def conversion_status(job_id):
    """
    Check status of MP4 conversion job.
    """
    if job_id not in conversion_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = conversion_jobs[job_id]

    response = {
        'progress': job['progress'],
        'complete': job['complete'],
        'error': job['error']
    }

    if job['complete']:
        response['downloadUrl'] = f'/api/download-mp4/{job_id}'

    return jsonify(response)


@app.route('/api/download-mp4/<job_id>', methods=['GET'])
def download_mp4(job_id):
    """
    Download converted MP4 file.
    """
    from flask import send_file

    if job_id not in conversion_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = conversion_jobs[job_id]

    if not job['complete']:
        return jsonify({'error': 'Conversion not complete'}), 400

    mp4_path = job['mp4_path']

    if not os.path.exists(mp4_path):
        return jsonify({'error': 'MP4 file not found'}), 404

    # Clean up job after download
    def cleanup():
        import time
        time.sleep(5)  # Wait for download to complete
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
        del conversion_jobs[job_id]

    import threading
    threading.Thread(target=cleanup).start()

    return send_file(mp4_path, mimetype='video/mp4', as_attachment=True,
                     download_name='uspsa-analysis-with-overlay.mp4')


# Global dict to track conversion jobs
conversion_jobs = {}


if __name__ == '__main__':
    if model is None:
        print("\n" + "="*50)
        print("⚠️  WARNING: Model not loaded!")
        print("="*50)
        print("Make sure shot_detector_model.h5 is in the current directory")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("🚀 USPSA Shot Detection Server")
        print("="*50)
        print(f"Model: {MODEL_PATH}")
        print(f"Threshold: {SHOT_THRESHOLD}")
        print("Server running on: http://localhost:5001")
        print("="*50 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5001)
