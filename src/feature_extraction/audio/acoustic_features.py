# acoustic features:

# phonation rate: total duration of speech to total duration of the sample (including pauses) (Fraser et al., 2014)
# total duration of speech: total length of all non-silent segments (Fraser et al., 2014; Tóth et al., 2017)
# speech rate phonemes: the number of total phonemes uttered, divided by the total duration of the utterance (Tóth et al., 2017; Huang et al., 2024; Gosztolya et al., 2019)
# speech rate words: the number of words divided by the duration of the speech sample (Wilson et al., 2010; Slegers et al., 2018)
# number of pauses: total number of pause occurrences (Tóth et al., 2017; Petti et al., 2020, but Fraser et al. (2014) pause length: 0.15 s)
# total length of pauses: total duration of pauses (as defined above) (Tóth et al., 2017) (in s instead of ms)
# average pause length: the total duration of pauses (as defined above) divided by the number of pauses (Tóth et al., 2017; Fraser et al., 2014)
# short pause count: number of pauses longer than 0.15 s but shorter than 0.4 s (Fraser et al., 2014)
# long pause count: number of pauses longer than 0.4 s (Fraser et al., 2014)
# pause / word ratio: ratio of silent segments longer than 0.15 s to non-silent segments (Fraser et al., 2014)
# pause ratio: the ratio of the total length of pauses to the total duration of the utterance (Tóth et al., 2017)
# pause rate: the number of pause occurrences divided by the total duration of the utterance (Tóth et al., 2017; Huang et al., 2024)

import eng_to_ipa as ipa
from feature_extraction.features import n_words
from feature_extraction.audio.voice_activity_detector import VoiceActivityDetector

vad = VoiceActivityDetector(debug=False)

def count_phonemes(text): # count IPA letters in transcriptions
    if isinstance(text, float) or text is None:
        text = ""
    phonemes = ipa.convert(text)
    return len([p for p in phonemes if p.isalpha()])

def extract_acoustic_features(audio_path, text, duration):
    # get VAD segments
    segments_df = vad.get_segments(audio_path)

    # calculate total voiced and pause durations
    total_voiced_duration = segments_df[segments_df["type"] == "voice"]["duration"].sum()
    total_pause_duration = segments_df[segments_df["type"] == "pause"]["duration"].sum()

    # count number of pauses (>= 150ms)
    pauses = segments_df[(segments_df["type"] == "pause") & (segments_df["duration"] >= 0.15)]
    n_pauses = pauses.shape[0]
    short_pauses = pauses[(pauses["duration"] < 0.4)]
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
        "avg_pause_duration": total_pause_duration / n_pauses if n_pauses > 0 else None,
        "short_pause_count": short_pauses.shape[0],
        "long_pause_count": long_pauses.shape[0],
        "pause_word_ratio": total_pause_duration / total_voiced_duration if word_count > 0 else None,
        "pause_ratio": total_pause_duration / duration if duration > 0 else None,
        "pause_rate": n_pauses / duration if duration > 0 else None,
    }

    return features
