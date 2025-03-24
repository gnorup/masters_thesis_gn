import parselmouth

file_path = "/var/folders/py/1h2n_yxd7l9dts6tjm4v1tpr0000gn/T/cookieTheft_converted.wav"

try:
    snd = parselmouth.Sound(file_path)
    print("✅ Praat can load the file!")
except Exception as e:
    print(f"❌ Praat error: {e}")

import webrtcvad
import librosa
import numpy as np
import soundfile as sf

audio_path = "/var/folders/py/1h2n_yxd7l9dts6tjm4v1tpr0000gn/T/cookieTheft_converted.wav"

# Load audio
y, sr = librosa.load(audio_path, sr=16000, mono=True)

# Convert to 16-bit PCM
y_int16 = (y * 32767).astype(np.int16)

# Frame size for WebRTC VAD
frame_length = 30  # 30ms per frame
frame_bytes = int(sr * (frame_length / 1000))  # Convert ms to samples

# Initialize VAD
vad = webrtcvad.Vad(3)  # Aggressive mode

# Check speech frames
speech_frames = sum(1 for i in range(0, len(y_int16) - frame_bytes, frame_bytes)
                    if vad.is_speech(y_int16[i:i+frame_bytes].tobytes(), sr))

print(f"✅ WebRTC detected {speech_frames} speech frames.")

import parselmouth
from parselmouth.praat import call

audio_path = "/var/folders/py/1h2n_yxd7l9dts6tjm4v1tpr0000gn/T/cookieTheft_converted.wav"

# Load audio in Praat
snd = parselmouth.Sound(audio_path)

# ✅ Fix: Add all required arguments for "To Pitch (cc)"
pitch = call(snd, "To Pitch (cc)", 0.0, 75, 500, 0.02, 0.3, 0.45, 0.01, 0.35, 0.14, 0.05)

# Get number of detected pitch points
num_pitch_points = call(pitch, "Get number of frames")

print(f"🟢 Number of detected pitch frames: {num_pitch_points}")

import parselmouth
from parselmouth.praat import call

audio_path = "/var/folders/py/1h2n_yxd7l9dts6tjm4v1tpr0000gn/T/cookieTheft_converted.wav"

snd = parselmouth.Sound(audio_path)

# ✅ Check if a PointProcess can be created
point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
num_points = call(point_process, "Get number of points")

print(f"🟢 Number of glottal pulses (PointProcess): {num_points}")

import parselmouth

audio_path = "/var/folders/py/1h2n_yxd7l9dts6tjm4v1tpr0000gn/T/cookieTheft_converted.wav"

# Load sound
snd = parselmouth.Sound(audio_path)

# ✅ List all available methods for Sound
print("\n🟢 Available attributes & methods for Sound object:")
for attr in dir(snd):
    print(attr)

# ✅ Create PointProcess and list its methods
point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

print("\n🟢 Available attributes & methods for PointProcess object:")
for attr in dir(point_process):
    print(attr)




import parselmouth
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load audio
audio_path = "/Users/gilanorup/Desktop/cookieTheft_converted.wav"
snd = parselmouth.Sound(audio_path)

# ✅ Extract amplitude values
time_values = np.linspace(snd.xmin, snd.xmax, snd.n_samples)
amplitude_values = snd.values.T.flatten()

# ✅ Plot waveform
plt.figure(figsize=(10, 5))
plt.plot(time_values, amplitude_values, color="black")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform of the Audio File")
plt.show()

from parselmouth.praat import call

# ✅ Compute Pitch with relaxed parameters
pitch = call(snd, "To Pitch (cc)", 0.0, 50, 400, 0.02, 0.3, 0.45, 0.01, 0.35, 0.14, 0.9)

# ✅ Count voiced frames
voiced_frames = sum(1 for i in range(call(pitch, "Get number of frames"))
                    if call(pitch, "Get value in frame", i + 1, "Hertz") is not None)

print(f"🟢 Number of Voiced Frames: {voiced_frames}")
