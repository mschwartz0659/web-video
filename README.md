# USPSA Shot Timer Analyzer

A web-based application for analyzing USPSA shooting videos using machine learning to automatically detect timer beeps and gunshots. Features an interactive interface with shot stream overlays and video export capabilities.

![USPSA Shot Timer Analyzer](https://img.shields.io/badge/ML-TensorFlow-orange) ![Platform](https://img.shields.io/badge/platform-web-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

### 🎯 ML-Powered Shot Detection
- Automatic detection of timer beep and gunshots using trained neural network
- 3-class classification model (non-shot, shot, beep)
- Real-time confidence scoring for each detection
- Adjustable sensitivity threshold

### 📊 Interactive Analysis
- **Waveform Visualization**: Audio waveform with marked shots and beep
- **Shot Stream Overlay**: Live overlay showing timer and split times
  - 4 position options: Top, Bottom, Left, Right
  - Dynamic sizing based on video dimensions
  - Smooth 60fps timer updates
- **Manual Editing**: Click on waveform to add/remove shot markers
- **Split Time Calculations**: Automatic calculation of draw time and shot splits

### 🎥 Video Export
- Export videos with embedded shot stream overlay
- Two format options:
  - **WebM**: Instant export (~30-60s) - VP9 codec
  - **MP4**: QuickTime compatible (~90-180s) - H.264 codec with server-side FFmpeg conversion
- Exports use original filename + `_uspsaml` suffix
- Progress tracking throughout export and conversion process

### 🎮 User Experience
- Custom video player controls
- Smooth seeking with waveform synchronization
- Responsive progress indicators during upload and processing
- Clean, modern UI built with Tailwind CSS

## Demo

1. Upload a USPSA shooting video
2. ML model automatically detects beep and shots
3. Review and edit detections on interactive waveform
4. Position shot stream overlay (top/bottom/left/right)
5. Export video with embedded overlay

## Technology Stack

### Frontend
- **HTML5** with Canvas API for waveform rendering
- **JavaScript** (vanilla) for application logic
- **Tailwind CSS** for styling
- **Web Audio API** for audio processing
- **MediaRecorder API** for video capture

### Backend
- **Flask** web server (Python)
- **TensorFlow/Keras** for ML inference
- **Librosa** for audio processing and feature extraction
- **FFmpeg** for video format conversion (MP4 export)
- **NumPy** for numerical computations

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **FFmpeg**: Required for MP4 export (install via Homebrew on macOS or apt on Linux)
- **Modern Browser**: Chrome, Firefox, or Safari with WebM support

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/web-video.git
cd web-video
```

### 2. Set Up Python Environment

**Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- flask
- flask-cors
- tensorflow
- librosa
- numpy
- soundfile

### 3. Model Files

Ensure the following files are present in the project root:
- `shot_detector_model.keras` or `shot_detector_model.h5` - Trained ML model
- `normalization_params.json` - Global normalization parameters for consistent detection

## Usage

### Starting the Server

```bash
python server.py
```

The server will start on `http://localhost:5001`

### Opening the Application

Open `index.html` in your web browser. You can either:
- Double-click the file to open in your default browser
- Serve it with a local web server:
  ```bash
  # Python 3
  python -m http.server 8000

  # Then visit http://localhost:8000
  ```

### Processing a Video

1. **Upload Video**: Click "Choose Video" and select your USPSA video file
   - Supported formats: MP4, MOV, AVI, WebM, MKV
   - Video processes in ~60 seconds on average hardware

2. **Review Results**:
   - Waveform shows detected beep (blue) and shots (red)
   - Shot stream overlay displays timer and split times
   - Verify all shots were detected correctly

3. **Edit if Needed**:
   - Click on waveform to add missing shots
   - Click existing markers to remove false positives
   - "Edited" badge appears when changes are made

4. **Position Shot Stream**:
   - Choose from Top, Bottom, Left, or Right position
   - Preview updates in real-time

5. **Export Video**:
   - Click "Export Video" button
   - Choose WebM (instant) or MP4 (with conversion)
   - Wait for processing to complete
   - Video downloads automatically with `_uspsaml` suffix

### Adjusting Detection Sensitivity

Use the sensitivity slider (0.1-0.9) to adjust shot detection:
- **Lower values** (0.3-0.5): More sensitive, may detect false positives
- **Higher values** (0.5-0.7): Less sensitive, may miss quiet shots
- **Default**: 0.5 (balanced)

## Configuration

### Server Configuration (`server.py`)

```python
# Port configuration
PORT = 5001  # Change if needed

# ML Model thresholds
SHOT_THRESHOLD = 0.5  # Confidence threshold for shots
BEEP_THRESHOLD = 0.5  # Confidence threshold for beep

# Audio processing parameters
SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.15  # Analysis window in seconds
N_MELS = 64  # Mel spectrogram resolution
```

### Client Configuration (`index.html`)

```javascript
// Server URL (line 176)
const SERVER_URL = 'http://localhost:5001';

// Change if running server on different host/port
```

## Project Structure

```
web-video/
├── index.html                      # Main application interface
├── server.py                       # Flask API server
├── requirements.txt                # Python dependencies
├── shot_detector_model.keras       # Trained ML model
├── normalization_params.json       # Model normalization parameters
├── README.md                       # This file
├── README-EXPORT.md               # Video export documentation
├── README-SERVER.md               # Server setup guide
├── PERFORMANCE-OPTIMIZATION.md    # Future optimization notes
├── training/                      # Training tools (optional)
│   ├── train_shot_detector.ipynb # Model training notebook
│   ├── labeling-tool.html        # Data labeling interface
│   ├── cleanup_nonshots.py       # Dataset cleanup utility
│   ├── debug_spectrograms.py     # Spectrogram visualization
│   └── test_model.py             # Model testing script
└── training_data/                 # Training dataset (not in git)
```

## API Endpoints

### POST `/api/analyze-video`
Analyze video for beep and shot detections.

**Request:**
- `video`: Video file (multipart/form-data)
- `threshold`: Optional detection threshold (float, 0.1-0.9)

**Response:**
```json
{
  "success": true,
  "beep_time": 5.23,
  "shots": [
    {"time": 6.45, "confidence": 0.87},
    {"time": 7.12, "confidence": 0.92}
  ],
  "total_shots": 2,
  "draw_time": 1.22
}
```

### POST `/api/convert-to-mp4`
Convert WebM to MP4 format.

**Request:**
- `video`: WebM video file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "job_id": "uuid-string"
}
```

### GET `/api/conversion-status/<job_id>`
Check MP4 conversion progress.

**Response:**
```json
{
  "complete": true,
  "progress": 100,
  "downloadUrl": "/api/download-mp4/<job_id>"
}
```

### GET `/api/download-mp4/<job_id>`
Download converted MP4 file.

## Troubleshooting

### Video Processing Issues

**Problem**: "No shots detected" or incorrect detections
- **Solution**: Adjust sensitivity slider and re-process
- **Cause**: Threshold may be too high/low for your video's audio levels

**Problem**: Processing takes longer than expected
- **Solution**: This is normal for longer videos or slower hardware
- **Note**: See `PERFORMANCE-OPTIMIZATION.md` for future speedup opportunities

### Export Issues

**Problem**: MP4 export fails with "ffprobe not found"
- **Solution**: Install FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- **Check**: Run `ffmpeg -version` to verify installation

**Problem**: Export video is corrupted or freezes
- **Solution**: Avoid seeking the video manually before export, or reload the page
- **Cause**: Event listener issues (should be fixed in latest version)

**Problem**: Exported video has wrong shot order
- **Solution**: Update to latest version - this was fixed in commit b520935

### Server Issues

**Problem**: Server won't start - "Address already in use"
- **Solution**: Kill process using port 5001: `lsof -ti:5001 | xargs kill -9`
- **Alternative**: Change PORT in `server.py`

**Problem**: Model loading fails
- **Solution**: Ensure `shot_detector_model.keras` or `.h5` file exists
- **Check**: File should be ~13MB in size

**Problem**: CORS errors in browser
- **Solution**: flask-cors is installed and enabled by default
- **Check**: Verify `CORS(app)` is present in `server.py`

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Firefox | ✅ Full | Works well |
| Safari | ✅ Full | macOS/iOS |
| Edge | ✅ Full | Chromium-based |
| IE 11 | ❌ No | Not supported |

## Performance

**Typical Processing Times** (on MacBook Pro M1):
- Upload: ~5-10 seconds (depends on file size)
- ML Analysis: ~60 seconds (for 30-60 second video)
- WebM Export: ~30-60 seconds
- MP4 Conversion: ~30 seconds

**Processing Time Factors:**
- Video length (longer = more segments to analyze)
- CPU speed (ML inference is CPU-bound)
- Available RAM (affects caching)
- Disk I/O speed (video decoding)

See `PERFORMANCE-OPTIMIZATION.md` for optimization opportunities (2-5x speedup possible).

## Training Your Own Model

If you want to train the model on your own data:

1. Use `training/labeling-tool.html` to label your videos
2. Organize labeled data in `training_data/` directory
3. Follow the Jupyter notebook `training/train_shot_detector.ipynb`
4. Replace `shot_detector_model.keras` and `normalization_params.json`

**Note**: Training requires labeled data and takes several hours on CPU.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Known Issues

- Very long videos (>5 minutes) may cause memory issues during export
- Safari may have issues with WebM export - use MP4 instead
- First video processing after server start may be slower (model loading)

## Roadmap

- [ ] Batch video processing
- [ ] Save/load detection sessions
- [ ] Custom shot stream styling options
- [ ] Real-time processing optimization (see PERFORMANCE-OPTIMIZATION.md)
- [ ] Mobile-responsive UI improvements
- [ ] Export presets for different social media platforms

## License

MIT License - See LICENSE file for details

## Acknowledgments

- TensorFlow team for the ML framework
- Librosa for audio processing capabilities
- FFmpeg for video conversion
- USPSA community for feedback and testing

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in `README-*.md` files
- Review `PERFORMANCE-OPTIMIZATION.md` for performance questions

## Version History

### v1.0.0 (Current)
- Initial release with ML-powered shot detection
- Interactive shot stream overlay with 4 positions
- WebM and MP4 export with embedded overlays
- Smooth UI with 60fps timer updates
- Comprehensive error handling and progress tracking

---

**Made with ❤️ for the USPSA shooting community**
