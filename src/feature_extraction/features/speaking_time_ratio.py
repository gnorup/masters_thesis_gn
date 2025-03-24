# percentage of recording that person is actively speaking

import librosa
import pandas as pd

def speaking_time_ratio(audio_path, total_duration):
    """
    Ratio of voiced time to total duration.
    """
    y, sr = librosa.load(audio_path, sr=16000)  # load waveform at 16kHz
    intervals = librosa.effects.split(y, top_db=30)  # identifies non-silent segments (segments that are over 30dB (-> reference?)); list of frame indices where speech happens

    intervals_df = pd.DataFrame(intervals, columns=['start', 'end'])

    # convert to seconds
    intervals_df = intervals_df / sr

    # calculate duration of each segment
    intervals_df['duration'] = intervals_df['end'] - intervals_df['start']

    voiced_duration = intervals_df['duration'].sum()
    return voiced_duration / total_duration if total_duration > 0 else None
