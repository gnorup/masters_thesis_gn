# Speech tempo: the number of phonemes per second during speech (including hesitations) (Tóth et al., 2017)

import eng_to_ipa as ipa # convert transcriptions to phonemes

def speech_tempo(text, duration):
    """Calculate speech tempo: number of phonemes per second (including hesitations)."""
    if isinstance(text, float):
        text = ""

    phonemes = ipa.convert(text)
    phoneme_count = len([p for p in phonemes if p.isalpha()])  # count IPA letters (skip spaces etc.)

    return phoneme_count / duration if duration > 0 else None
