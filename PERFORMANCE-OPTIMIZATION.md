# Performance Optimization Ideas

## Current Processing Time
- Average: ~60 seconds for a typical USPSA video
- Bottleneck: ML shot detection scanning entire video

## Identified Optimization Opportunity

### Problem
The main performance bottleneck is in `server.py` line 89:
```python
y, _ = librosa.load(video_path, sr=sr, offset=timestamp, duration=duration, mono=True)
```

This is called for **every segment** during scanning (typically 200-400+ times per video at 0.05s intervals). Each call:
- Decodes the entire video file
- Seeks to the timestamp
- Extracts a 0.15s segment

This means the video is decoded hundreds of times, which is extremely inefficient.

### Proposed Solution: Pre-load Audio
Load the entire audio once at the start, then extract segments from the in-memory array:

```python
def scan_video_for_shots_and_beeps(video_path, scan_interval=0.05, shot_threshold=None, beep_threshold=None):
    # Load entire audio once
    full_audio, sr = librosa.load(video_path, sr=SAMPLE_RATE, mono=True)
    duration = len(full_audio) / sr

    # Then extract segments from the array
    for timestamp in scan_times:
        start_sample = int(timestamp * sr)
        end_sample = int((timestamp + SEGMENT_DURATION) * sr)
        segment = full_audio[start_sample:end_sample]

        # Process segment (convert to mel spectrogram, etc.)
        ...
```

### Expected Performance Gain
- **2-5x faster** processing on same hardware
- Primarily I/O savings, not changing ML accuracy
- On production hardware with faster CPU/GPU: could be 5-10x faster than current

### Implementation Considerations

**Pros:**
- Pure I/O optimization - ML model sees identical data
- No accuracy loss
- Significant speedup

**Cons:**
- Need to load entire audio into memory (typically 5-20MB for a 30-60s video)
- Requires refactoring `extract_audio_segment()` to accept pre-loaded audio
- Need to handle edge cases (end of file, etc.)
- **Must test thoroughly** to ensure segment extraction produces identical spectrograms

### Testing Required Before Deployment
1. Process same video with both methods
2. Compare spectrogram outputs to ensure they're identical
3. Compare detection results (beep time, shot times, confidences)
4. Test with various video lengths and formats
5. Monitor memory usage for long videos

### Alternative Optimizations (Lower Impact)

1. **Batch Predictions** (~10-20% faster)
   - Group segments and predict in batches instead of one-by-one
   - Pure TensorFlow optimization

2. **GPU Acceleration** (2-3x faster on GPU)
   - Ensure TensorFlow uses GPU/MPS if available
   - Check `tf.config.list_physical_devices('GPU')`

3. **Increase scan_interval** (faster but may miss shots)
   - Current: 0.05s (50ms intervals)
   - Could try: 0.075s or 0.1s
   - ⚠️ Risk: Might miss shots that occur between scan points

## Decision
**Status**: Deferred

**Rationale**:
- Current implementation prioritizes correctness and is working well
- Processing time (~60s) is acceptable for non-real-time analysis
- Production hardware will naturally be faster
- Risk of introducing bugs that affect detection accuracy
- Can revisit if processing time becomes a blocker

**When to Revisit**:
- Processing videos in production at scale
- User feedback about slow processing
- After deployment on production hardware (to measure real-world impact)
- If processing many videos in batch
