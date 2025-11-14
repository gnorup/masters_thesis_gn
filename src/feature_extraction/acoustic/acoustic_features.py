import eng_to_ipa as ipa
from feature_extraction.linguistic import n_words
from feature_extraction.acoustic.voice_activity_detector import VoiceActivityDetector

# Voice activity detector (WebRTCVAD-based)
vad = VoiceActivityDetector(debug=False)

def count_phonemes(text):
    """
    Count IPA phoneme symbols in the transcript using eng_to_ipa.
    """
    if isinstance(text, float) or text is None:
        text = ""
    phonemes = ipa.convert(text)
    return len([p for p in phonemes if p.isalpha()])

def extract_acoustic_features(audio_path, text, duration):
    """
    Extract acoustic fluency measures.
    Feature definitions based on Fraser et al. (2014), Tóth et al. (2018), Huang et al. (2024), Slegers et al. (2018),
    Petti et al. (2020), Gosztolya et al. (2019), Wilson et al. (2010).
    """

    # get VAD segments
    segments_df = vad.get_segments(audio_path)

    # calculate total voiced and pause durations
    voiced_segments = segments_df[segments_df["type"] == "voice"]
    pause_segments = segments_df[segments_df["type"] == "pause"]

    total_voiced_duration = voiced_segments["duration"].sum()
    total_pause_duration = pause_segments["duration"].sum()

    # count number of pauses (>= 150ms)
    pauses = pause_segments[pause_segments["duration"] >= 0.15]
    n_pauses = pauses.shape[0]
    long_pauses = pauses[(pauses["duration"] >= 0.4)]

    # word and phoneme count from transcript
    word_count = n_words(text)
    phoneme_count = count_phonemes(text)

    # feature dictionary
    features = {
        "phonation_rate": total_voiced_duration / duration if duration > 0 else None,
        "total_speech_duration": total_voiced_duration,
        "speech_rate_phonemes": phoneme_count / duration if duration > 0 else None,
        "speech_rate_words": word_count / duration if duration > 0 else None,
        "n_pauses": n_pauses,
        "total_pause_duration": total_pause_duration,
        "avg_pause_duration": total_pause_duration / n_pauses if n_pauses > 0 else 0.0,
        "long_pause_count": long_pauses.shape[0],
        "pause_word_ratio": total_pause_duration / total_voiced_duration if word_count > 0 else None,
        "pause_ratio": total_pause_duration / duration if duration > 0 else None,
        "pause_rate": n_pauses / duration if duration > 0 else None,
    }

    return features