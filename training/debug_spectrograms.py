"""
Diagnostic script to compare spectrograms from training vs inference
This will help identify why the model works on exact timestamps but not sliding windows
"""
import numpy as np
import librosa

# Same parameters as training and inference
SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.15
N_MELS = 64
SPEC_SIZE = 64

def extract_audio_segment(video_path, timestamp, duration=SEGMENT_DURATION, sr=SAMPLE_RATE):
    """
    Extract audio segment - EXACT SAME as server.py
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

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to fixed dimensions
        if mel_spec_db.shape[1] < SPEC_SIZE:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, SPEC_SIZE - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :SPEC_SIZE]

        # Normalize to 0-1 range (MUST match training!)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        return mel_spec_db, y

    except Exception as e:
        print(f"Error extracting audio at {timestamp}s: {e}")
        return None, None


# Test video
video_path = 'training_data/stage6a.mov'

print("=" * 70)
print("🔍 SPECTROGRAM DIAGNOSTIC - Comparing Training vs Inference")
print("=" * 70)

# Exact labeled shot times from Step 13
labeled_shots = [0.20, 0.80, 1.40, 2.00, 2.60, 3.10, 3.60, 4.10, 4.60, 5.10]

# Random times from sliding window (NOT labeled shots)
random_times = [0.50, 1.00, 1.50, 2.50, 5.00, 10.00, 15.00, 20.00]

print("\n1️⃣ TESTING LABELED SHOT TIMESTAMPS (Model should detect these)")
print("-" * 70)

shot_specs = []
for i, timestamp in enumerate(labeled_shots[:5]):  # Test first 5
    spec, audio = extract_audio_segment(video_path, timestamp)
    if spec is not None:
        shot_specs.append(spec)
        print(f"  Shot @ {timestamp}s:")
        print(f"    Audio samples: {len(audio)}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")
        print(f"    Spec min: {spec.min():.4f}, max: {spec.max():.4f}, mean: {spec.mean():.4f}")
        print(f"    Spec std: {spec.std():.4f}")

print("\n2️⃣ TESTING RANDOM TIMESTAMPS (Model should NOT detect these as shots)")
print("-" * 70)

random_specs = []
for timestamp in random_times:
    spec, audio = extract_audio_segment(video_path, timestamp)
    if spec is not None:
        random_specs.append(spec)
        print(f"  Random @ {timestamp}s:")
        print(f"    Audio samples: {len(audio)}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")
        print(f"    Spec min: {spec.min():.4f}, max: {spec.max():.4f}, mean: {spec.mean():.4f}")
        print(f"    Spec std: {spec.std():.4f}")

print("\n3️⃣ STATISTICAL COMPARISON")
print("-" * 70)

if shot_specs and random_specs:
    shot_specs = np.array(shot_specs)
    random_specs = np.array(random_specs)

    print(f"Shot spectrograms:")
    print(f"  Mean value: {shot_specs.mean():.4f}")
    print(f"  Std dev: {shot_specs.std():.4f}")
    print(f"  Min: {shot_specs.min():.4f}, Max: {shot_specs.max():.4f}")

    print(f"\nRandom spectrograms:")
    print(f"  Mean value: {random_specs.mean():.4f}")
    print(f"  Std dev: {random_specs.std():.4f}")
    print(f"  Min: {random_specs.min():.4f}, Max: {random_specs.max():.4f}")

    print(f"\n🔍 ANALYSIS:")
    mean_diff = abs(shot_specs.mean() - random_specs.mean())
    print(f"  Mean difference: {mean_diff:.4f}")

    if mean_diff < 0.1:
        print("  ❌ PROBLEM FOUND: Shot and random spectrograms look too similar!")
        print("     This explains why the model classifies everything as shots.")
        print("     The normalization (min-max per segment) makes all spectrograms")
        print("     look similar regardless of actual audio content.")
    else:
        print("  ✅ Spectrograms show clear difference")

print("\n4️⃣ TESTING WITH MODEL")
print("-" * 70)

try:
    import tensorflow as tf
    model = tf.keras.models.load_model('shot_detector_model.h5', compile=False)
    print("✅ Model loaded")

    print("\nPredictions on SHOT timestamps:")
    for i, spec in enumerate(shot_specs[:5]):
        spec_input = spec.reshape(1, N_MELS, SPEC_SIZE, 1)
        pred = model.predict(spec_input, verbose=0)[0][0]
        print(f"  Shot {i+1} @ {labeled_shots[i]}s: {pred:.4f}")

    print("\nPredictions on RANDOM timestamps:")
    for i, spec in enumerate(random_specs):
        spec_input = spec.reshape(1, N_MELS, SPEC_SIZE, 1)
        pred = model.predict(spec_input, verbose=0)[0][0]
        print(f"  Random @ {random_times[i]}s: {pred:.4f}")

except Exception as e:
    print(f"❌ Could not load model: {e}")

print("\n" + "=" * 70)
print("🎯 DIAGNOSIS COMPLETE")
print("=" * 70)
