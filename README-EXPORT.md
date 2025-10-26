# Video Export System

## Overview

The USPSA Shot Timer Analyzer supports hybrid video export:
1. **Instant WebM** export (browser-based rendering)
2. **Optional MP4** conversion (server-based with FFmpeg)

## Export Process

### Step 1: WebM Export (Instant)
- Click "Export Video" button
- Video renders in browser at 30fps with shot streamer overlay
- Downloads immediately as `.webm` file (~30-60 seconds)
- **Format**: WebM (VP9 video codec + Opus audio)
- **Quality**: 8 Mbps bitrate

### Step 2: MP4 Conversion (Optional)
After WebM download, a dialog appears with two options:

**Option A: "Done"**
- Use WebM file directly
- Works in: VLC, Chrome, Firefox, Edge, most modern players
- **Not compatible**: QuickTime, Windows Media Player

**Option B: "Convert to MP4"**
- Uploads WebM to server
- Converts to true MP4 using FFmpeg
- Downloads MP4 file (~1-2 minutes total)
- **Format**: MP4 (H.264 video + AAC audio)
- **Compatible with**: QuickTime, all players

## Technical Details

### WebM Export
- **Rendering**: Client-side via Canvas API + MediaRecorder
- **Overlay**: Shot streamer with countdown timer and split times
- **Timer behavior**:
  - Shows negative countdown before beep (-X.XX)
  - Shows positive timer after beep (+X.XX)
  - Updates in real-time as video plays

### MP4 Conversion
- **Server**: Flask endpoint at `POST /api/convert-to-mp4`
- **Conversion**: FFmpeg with H.264/AAC codecs
- **Progress tracking**: Polls server every second
- **Cleanup**: Temporary files auto-deleted after download

## Features

### Both Formats Include:
- Shot streamer overlay in selected position (top/bottom/left/right)
- Countdown timer before beep
- Draw time and split times
- Timer + last 4 shots visible at any time
- Responsive sizing based on video resolution

### Quality Settings:
- **WebM**: 8 Mbps, VP9 codec, 30fps
- **MP4**: CRF 23 (high quality), H.264, 192k audio, faststart enabled

## Prerequisites

### For WebM Export:
- Modern browser (Chrome, Firefox, Edge, Safari)
- No server required

### For MP4 Conversion:
- FFmpeg installed on server
- Flask server running on port 5001

## Installation

### Install FFmpeg (macOS):
```bash
brew install ffmpeg
```

### Install FFmpeg (Ubuntu/Debian):
```bash
sudo apt-get install ffmpeg
```

### Verify Installation:
```bash
ffmpeg -version
ffprobe -version
```

## Usage

1. Analyze video and get shot detection results
2. Position shot streamer (top/bottom/left/right)
3. Click "Export Video"
4. Wait for WebM download (30-60 seconds)
5. Choose:
   - Click "Done" to keep WebM
   - Click "Convert to MP4" to get MP4 (additional 1-2 minutes)

## Troubleshooting

### WebM Export Issues

**Problem**: Progress bar stuck at 99%
- **Solution**: Fixed - progress now updates every 0.5 seconds with current time

**Problem**: No countdown timer before beep
- **Solution**: Fixed - timer now shows from start of video

**Problem**: Overlay not visible
- **Solution**: Check that shots were detected, verify streamer position

### MP4 Conversion Issues

**Problem**: "Converting on server..." then stops
- **Cause**: FFmpeg not installed on server
- **Solution**: Install FFmpeg (see Installation section above)

**Problem**: Conversion fails with error
- **Check server logs** for FFmpeg error messages
- Verify WebM file is valid
- Ensure sufficient disk space in temp directory

**Problem**: Download doesn't start
- Check browser console for errors
- Verify server is running on correct port (5001)
- Check CORS settings

## File Formats

### WebM (Browser-rendered)
- **Extension**: `.webm`
- **Container**: Matroska/WebM
- **Video**: VP9
- **Audio**: Opus
- **Size**: ~1-2 MB per minute

### MP4 (Server-converted)
- **Extension**: `.mp4`
- **Container**: MP4
- **Video**: H.264 (libx264)
- **Audio**: AAC
- **Size**: Similar to WebM (~1-2 MB per minute)

## Performance

### WebM Export:
- **Time**: 30-60 seconds (real-time rendering)
- **Network**: None (client-side only)
- **Server load**: None

### MP4 Conversion:
- **Upload**: 5-15 seconds (depends on connection)
- **Conversion**: 10-30 seconds (FFmpeg processing)
- **Download**: 5-15 seconds
- **Total**: ~90-180 seconds

## Server API

### POST /api/convert-to-mp4
Upload WebM file for MP4 conversion.

**Request:**
- `video`: WebM file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "jobId": "uuid",
  "statusUrl": "/api/conversion-status/uuid"
}
```

### GET /api/conversion-status/:jobId
Check conversion progress.

**Response:**
```json
{
  "progress": 75,
  "complete": false,
  "error": null,
  "downloadUrl": null
}
```

### GET /api/download-mp4/:jobId
Download converted MP4 file.

**Response:** MP4 file stream

## Future Enhancements

- [ ] Batch export multiple positions
- [ ] Custom overlay text/branding
- [ ] Direct MP4 export (bypass WebM step)
- [ ] Cloud-based conversion service
- [ ] Export queue for multiple videos
