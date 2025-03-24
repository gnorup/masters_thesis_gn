# percentage of recording that person is actively speaking

import librosa

def speaking_time_ratio(audio_path, total_duration):
    """
    Ratio of voiced time to total duration.
    """
    y, sr = librosa.load(audio_path, sr=16000) # load waveform at 16kHz
    intervals = librosa.effects.split(y, top_db=30) # identifies non-silent segments (segments that are over 30dB (-> reference?)); list of frame indices where speech happens

    voiced_duration = sum((end - start) for start, end in intervals) / sr # number of samples in speech segment, divide by sample rate to convert to seconds -> sum: total time spent speaking
    return voiced_duration / total_duration if total_duration > 0 else None
