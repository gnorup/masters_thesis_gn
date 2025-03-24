# shimmer and jitter are apparently impossible for this type of audiofile
# jitter: frequency variation

import parselmouth
from parselmouth.praat import call

def jitter(audio_path):
    """Compute jitter using Praat via Parselmouth."""
    try:
        snd = parselmouth.Sound(audio_path)  # load audio
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)  # extract voice periods
        jitter_value = call([snd, point_process], "Get jitter (local)", 0.0001, 0.02, 1.3, 1.6, 0.03, 1.3) # parameters: minimum pitch period (s), maximum pitch period (s), max period factor, max amplitude factor, silence threshold, voicing threshold

        return jitter_value
    except Exception as e:
        # print(f"Error processing jitter for {audio_path}: {e}")
        return None
