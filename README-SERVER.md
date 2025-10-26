# 🚀 USPSA Shot Detection Server - Local Testing Guide

This guide helps you run the ML-powered shot detection server on your local machine.

---

## 📋 Prerequisites

- **Python 3.9 or higher** installed on your machine
- **The trained model file**: `shot_detector_model.h5` (should already be in this directory)
- **A web browser** (Chrome, Firefox, Safari, etc.)

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Python Dependencies

Open terminal in this directory and run:

```bash
pip install -r requirements.txt
```

**What this installs:**
- Flask (web server)
- TensorFlow (ML model)
- librosa (audio processing)
- Other dependencies

**Note:** Installation may take 5-10 minutes depending on your internet connection.

---

### Step 2: Start the Server

```bash
python server.py
```

**You should see:**
```
==================================================
🚀 USPSA Shot Detection Server
==================================================
Model: shot_detector_model.h5
Threshold: 0.5
Server running on: http://localhost:5000
==================================================

 * Running on http://127.0.0.1:5000
```

**If you see an error about the model not loading:**
- Make sure `shot_detector_model.h5` is in the same directory as `server.py`
- Check the file isn't corrupted

**Keep this terminal window open** - the server needs to stay running!

---

### Step 3: Open the Web App

1. Open a **new browser tab**
2. Navigate to the `index-ml.html` file:
   - **Mac**: Right-click → Open With → Browser
   - **Windows**: Right-click → Open with → Browser
   - **Or** drag the file into your browser

3. You should see: **"🤖 USPSA Shot Timer Analyzer - ML Powered"**

---

### Step 4: Test with a Video

1. Click **"Select Video"**
2. Choose one of your USPSA stage videos
3. Wait for processing (~30-60 seconds depending on video length)
4. View results:
   - Detected shots on waveform
   - Draw time, stage time, split times
   - ML confidence scores

---

## 🎯 How It Works

```
User Browser (index-ml.html)
         ↓
    Upload video
         ↓
Flask Server (server.py)
         ↓
   1. Extract audio
   2. Scan with ML model
   3. Detect shots
   4. Detect beep
         ↓
    Return JSON results
         ↓
User Browser displays results
```

---

## 🔧 Troubleshooting

### Server won't start

**Error: "Model not loaded"**
- Check `shot_detector_model.h5` is in the directory
- File size should be ~10-20MB

**Error: "Address already in use"**
- Port 5000 is already taken
- Edit `server.py`, change port: `app.run(port=5001)`
- Update URL in `index-ml.html` to `http://localhost:5001`

**Error: "Module not found"**
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.9+

---

### Video upload fails

**Error: "Server error"**
- Check the server terminal for error messages
- Try a shorter video (<2 minutes) first
- Make sure video format is supported (MP4, MOV, AVI, WebM, MKV)

**Upload takes forever**
- Large videos (>500MB) take longer
- Try compressing the video first
- Or use a shorter clip for testing

---

### Results look wrong

**Too many shots detected (false positives)**
- Edit `server.py`, increase `SHOT_THRESHOLD` from 0.5 to 0.6 or 0.7
- Higher threshold = fewer detections

**Too few shots detected (missing shots)**
- Edit `server.py`, decrease `SHOT_THRESHOLD` from 0.5 to 0.4
- Lower threshold = more detections
- Remember: model has 100% recall on training data, so this is unlikely

**Beep detection wrong**
- Beep uses the original algorithm, not ML
- If consistently wrong, we can tune the beep detection parameters

---

## 📊 Understanding the Results

### ML Confidence Scores

Each detected shot has a confidence score (0-100%):
- **90-100%**: Very confident (definitely a shot)
- **70-90%**: Confident (likely a shot)
- **50-70%**: Uncertain (might be other loud sound)

Shots with lower confidence might be:
- Background shots from other stages
- Loud reloads or magazine drops
- Steel plate hits

### Markers on Waveform

- **Blue line**: Timer beep (start signal)
- **Green lines**: ML-detected shots
- **Brighter green**: Higher confidence
- **Red dashed line**: Video playhead (while playing)

---

## 🎛️ Adjusting Parameters

### In `server.py`:

```python
# Adjust detection threshold (line ~20)
SHOT_THRESHOLD = 0.5  # Try 0.6 for fewer detections, 0.4 for more

# Adjust scan interval (affects speed vs accuracy)
scan_interval=0.05  # In scan_video_for_shots() function
# 0.05 = scan every 50ms (current, good balance)
# 0.1 = faster but might miss some shots
# 0.02 = slower but more thorough
```

After changing parameters:
1. Stop the server (Ctrl+C in terminal)
2. Restart: `python server.py`
3. Reload the web page
4. Test again

---

## 📈 Performance Tips

### For faster processing:

1. **Use shorter videos** for testing
2. **Increase scan_interval** to 0.1 (faster, less accurate)
3. **Use GPU** if you have one (TensorFlow will auto-detect)

### For better accuracy:

1. **Decrease scan_interval** to 0.02 (slower, more accurate)
2. **Adjust SHOT_THRESHOLD** based on your videos

---

## 🐛 Debug Mode

To see detailed processing logs:

The server already runs in debug mode. Check the terminal where `server.py` is running to see:
- Video filename being processed
- Number of detections found
- Beep detection results
- Any errors

Example output:
```
==================================================
Processing: stage5_glasses.mp4
==================================================

🔍 Step 1: Scanning for shots...
  Progress: 0/1000 (0.0%)
  Progress: 100/1000 (10.0%)
  ...
✅ Found 26 potential shots
✅ After merging: 24 shots

🔔 Step 2: Detecting timer beep...
  Beep detected at: 2.84s

📊 Results:
  Beep: 2.84s
  Shots detected: 24
  Draw time: 1.52s
==================================================
```

---

## 🔄 Testing Workflow

### Good testing approach:

1. **Test with known video first**
   - Use one from your training set
   - You know the expected shot count
   - Verify model works correctly

2. **Test with new video**
   - Not in training set
   - Real-world test of model

3. **Compare with manual count**
   - Play video and count shots manually
   - Compare with ML detected count
   - Tune threshold if needed

4. **Test different scenarios**
   - First-person video
   - Third-person video
   - Noisy environment
   - Clean recording

---

## 📝 Next Steps

Once local testing works well:

1. **Share feedback** - what's working, what's not
2. **Tune parameters** - adjust threshold, scan interval
3. **Test on multiple videos** - verify consistency
4. **Plan cloud deployment** - choose hosting service
5. **Build production frontend** - polish the UI

---

## ❓ Questions?

Common questions:

**Q: Can I use this offline?**
A: Yes! Everything runs locally, no internet needed (after installation).

**Q: How much RAM does it need?**
A: ~2-4GB for the model + video processing.

**Q: Can I process multiple videos at once?**
A: Not yet, but we can add batch processing.

**Q: Will my videos be stored?**
A: No, they're deleted immediately after processing.

---

## 🎉 You're Ready!

Start the server and test it out. Let me know:
- How accurate the detections are
- If you need to adjust any parameters
- Any bugs or issues you find

Happy testing! 🚀
