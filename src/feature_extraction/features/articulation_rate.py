from feature_extraction.features import n_words
import librosa

def articulation_rate(text, audio_path):
    """
    Words per second, excluding pauses (based on voiced segments only)
    """
    word_count = n_words(text)

    # load audio at 16kHz
    y, sr = librosa.load(audio_path, sr=16000)

    # detect voiced segments (non-silent regions)
    intervals = librosa.effects.split(y, top_db=30)

    # compute total duration of voiced segments in seconds
    voiced_duration = sum((end - start) for start, end in intervals) / sr # converts frame indices to seconds, adds up all durations where speech was detected

    return word_count / voiced_duration if voiced_duration > 0 else None 
