# should be phonemes per second but it's an approximation

import librosa # library for audio / speech analysis

def speech_tempo(audio_path):
    """Calculate speech tempo: number of phonemes per second (including hesitations)."""
    y, sr = librosa.load(audio_path, sr=16000)  # load audio at 16kHz (y = waveform; sr = sampling rate)
    duration = librosa.get_duration(y=y, sr=sr)  # calculate total duration in seconds

    phoneme_count = len(librosa.effects.split(y, top_db=30))  # approximate phonemes -> returns segments of the audio where the volume is above a silence threshold (each non-silent region -> phoneme-like unit)
    speech_tempo_value = phoneme_count / duration if duration > 0 else None

    return speech_tempo_value
