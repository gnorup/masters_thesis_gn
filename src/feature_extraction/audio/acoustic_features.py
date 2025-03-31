# ideas for features: Tóth et al. (2017)

from feature_extraction.features import n_words
from feature_extraction.audio.voice_activity_detector import VoiceActivityDetector

vad = VoiceActivityDetector(debug=False)

def acoustic_features(audio_path, text, duration):
    """
    Extracts acoustic features based on VAD:
    - pause ratio
    - number of pauses
    - average pause duration
    - articulation rate
    - hesitation ratio (based on pauses > 300ms and short voiced segments < 600ms)
    """
    segments_df = vad.get_segments(audio_path)

    voiced_duration = segments_df[segments_df["type"] == "voice"]["duration"].sum()
    pause_duration = segments_df[segments_df["type"] == "pause"]["duration"].sum()
    n_pauses = segments_df[segments_df["type"] == "pause"].shape[0]
    word_count = n_words(text)

    # Identify hesitation-like segments
    long_pauses = segments_df[(segments_df["type"] == "pause") & (segments_df["duration"] >= 0.3)]
    short_voiced = segments_df[(segments_df["type"] == "voice") & (segments_df["duration"] <= 0.6)]

    hesitation_time = long_pauses["duration"].sum() + short_voiced["duration"].sum()

    features = {
        "pause_ratio": pause_duration / duration if duration > 0 else None,
        "n_pauses": n_pauses,
        "avg_pause_duration": pause_duration / n_pauses if n_pauses > 0 else None,
        "articulation_rate": word_count / voiced_duration if voiced_duration > 0 else None,
        "hesitation_ratio": hesitation_time / duration if duration > 0 else None
    }

    return features

