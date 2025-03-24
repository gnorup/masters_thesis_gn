# percentage of recording that person is not actively speaking

import librosa

def pause_ratio(audio_path, total_duration):
    y, sr = librosa.load(audio_path, sr=16000)
    intervals = librosa.effects.split(y, top_db=30)
    voiced_duration = sum((end - start) for start, end in intervals) / sr
    return 1 - (voiced_duration / total_duration) if total_duration > 0 else None # basically 1 - speaking_time_ratio
