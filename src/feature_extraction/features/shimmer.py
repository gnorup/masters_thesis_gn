# shimmer and jitter are apparently impossible for this type of audiofile
# shimmer: amplitude variation

import parselmouth # python interface for praat
from parselmouth.praat import call # call praat commands

def shimmer(audio_path):
    """Compute shimmer using Praat via Parselmouth."""
    try:
        snd = parselmouth.Sound(audio_path) # load file into sound object that praat can work with
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500) # create point process (sequence of glottal pulse events, pitch periods); 75 and 500 as expected pitch range in Hz
        if not point_process:
            raise ValueError("Point process extraction failed (no voiced frames).")

        shimmer_value = call([snd, point_process], "Get shimmer (local)", 0.0001, 0.02, 1.3, 1.6, 0.03, 1.6) # parameters: minimum pitch period (s), maximum pitch period (s), max period factor, max amplitude factor, silence threshold, voicing threshold

        return shimmer_value if shimmer_value > 0 else None  # prevent returning invalid values
    except Exception as e:
        print(f"Error processing shimmer for {audio_path}: {e}")
        return None
