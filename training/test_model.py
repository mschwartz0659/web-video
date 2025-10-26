"""
Quick test to see if the model is actually making predictions correctly
"""
import numpy as np
import tensorflow as tf

# Load the model
print("Loading model...")
model = tf.keras.models.load_model('shot_detector_model_old.h5', compile=False)
print(f"Model loaded: {model.input_shape} -> {model.output_shape}")

# Create test inputs
print("\nTesting with different inputs:")

# Test 1: All zeros (silence)
zeros = np.zeros((1, 64, 64, 1))
pred_zeros = model.predict(zeros, verbose=0)[0][0]
print(f"  All zeros (silence): {pred_zeros:.4f}")

# Test 2: All ones (loud noise)
ones = np.ones((1, 64, 64, 1))
pred_ones = model.predict(ones, verbose=0)[0][0]
print(f"  All ones (loud): {pred_ones:.4f}")

# Test 3: Random noise
random_noise = np.random.randn(1, 64, 64, 1)
pred_random = model.predict(random_noise, verbose=0)[0][0]
print(f"  Random noise: {pred_random:.4f}")

# Test 4: Another random sample
random_noise2 = np.random.randn(1, 64, 64, 1)
pred_random2 = model.predict(random_noise2, verbose=0)[0][0]
print(f"  Random noise 2: {pred_random2:.4f}")

print("\n" + "="*50)
if abs(pred_zeros - pred_ones) < 0.01 and abs(pred_zeros - pred_random) < 0.1:
    print("❌ MODEL IS BROKEN!")
    print("   All inputs produce similar predictions")
    print("   The model didn't learn anything during training")
else:
    print("✅ Model seems to be working")
    print("   Different inputs produce different predictions")
print("="*50)
