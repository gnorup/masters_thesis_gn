import librosa # loads audio
import numpy as np # for efficient energy calculations

def hesitation_ratio(audio_path):
    """Compute hesitation ratio based on silence detection."""

    y, sr = librosa.load(audio_path, sr=16000, mono=True) # load audio file, mono=True ensures it's a single-channel waveform

    # compute short-term energy
    frame_length = int(sr * 0.03)  # 30ms per frame -> loops over audio in steps of 480 samples
    energy = np.array([sum(abs(y[i:i+frame_length] ** 2)) for i in range(0, len(y), frame_length)]) # calculates short-term energy for each frame

    silence_threshold = np.percentile(energy, 25)  # define silence as bottom 25% energy frames (-> reference?)
    silence_frames = sum(energy < silence_threshold) # count how many frames are below threshold
    total_frames = len(energy) # compare to total number of frames

    print(f"Silence frames: {silence_frames}, Total frames: {total_frames}")

    if total_frames > 0:
        return silence_frames / total_frames  # ratio of silence frames
    else:
        return None
