# lists of possible features

def get_linguistic_features():
    return {
    "n_words","ttr","mattr_10","mattr_20","mattr_30","mattr_40","mattr_50","filler_word_ratio",
    "average_word_length","brunets_index","honores_statistic","guirauds_statistic","light_verb_ratio",
    "empty_word_ratio","nid_ratio","adjacent_repetitions","aoa_content","aoa_nouns","aoa_verbs",
    "fam_content","fam_nouns","fam_verbs","img_content","img_nouns","img_verbs","freq_content",
    "freq_nouns","freq_verbs","concr_content","concr_nouns","concr_verbs","um_ratio","uh_ratio",
    "er_ratio","ah_ratio","ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
    "PRON","PROPN","SCONJ","VERB","OTHER","NOUN/VERB","PRON/NOUN","DET/NOUN","AUX/VERB",
    "OPEN/CLOSED","information_words","article_pause_contentword"
    }

def get_acoustic_features():
    return {
        "phonation_rate","total_speech_duration","speech_rate_phonemes","speech_rate_words","n_pauses",
        "total_pause_duration","avg_pause_duration","short_pause_count","long_pause_count","pause_word_ratio",
        "pause_ratio","pause_rate",
        "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_amean","eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile20.0","eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
        "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile80.0","eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
        "eGeMAPS_loudness_sma3_amean","eGeMAPS_loudness_sma3_stddevNorm","eGeMAPS_loudness_sma3_percentile20.0",
        "eGeMAPS_loudness_sma3_percentile50.0","eGeMAPS_loudness_sma3_percentile80.0","eGeMAPS_loudness_sma3_pctlrange0-2",
        "eGeMAPS_loudness_sma3_meanRisingSlope","eGeMAPS_loudness_sma3_stddevRisingSlope",
        "eGeMAPS_loudness_sma3_meanFallingSlope","eGeMAPS_loudness_sma3_stddevRisingSlope",
        "eGeMAPS_spectralFlux_sma3_amean","eGeMAPS_spectralFlux_sma3_stddevNorm",
        "eGeMAPS_mfcc1_sma3_amean","eGeMAPS_mfcc1_sma3_stddevNorm","eGeMAPS_mfcc2_sma3_amean",
        "eGeMAPS_mfcc2_sma3_stddevNorm","eGeMAPS_mfcc3_sma3_amean","eGeMAPS_mfcc3_sma3_stddevNorm",
        "eGeMAPS_mfcc4_sma3_amean","eGeMAPS_mfcc4_sma3_stddevNorm",
        "eGeMAPS_jitterLocal_sma3nz_amean","eGeMAPS_jitterLocal_sma3nz_stddevNorm",
        "eGeMAPS_shimmerLocaldB_sma3nz_amean","eGeMAPS_shimmerLocaldB_sma3nz_stddevNorm",
        "eGeMAPS_HNRdBACF_sma3nz_amean","eGeMAPS_HNRdBACF_sma3nz_stddevNorm",
        "eGeMAPS_logRelF0-H1-H2_sma3nz_amean","eGeMAPS_logRelF0-H1-H2_sma3nz_stddevNorm",
        "eGeMAPS_logRelF0-H1-A3_sma3nz_amean","eGeMAPS_logRelF0-H1-A3_sma3nz_stddevNorm",
        "eGeMAPS_F1frequency_sma3nz_amean","eGeMAPS_F1frequency_sma3nz_stddevNorm",
        "eGeMAPS_F1bandwidth_sma3nz_amean","eGeMAPS_F1bandwidth_sma3nz_stddevNorm",
        "eGeMAPS_F1amplitudeLogRelF0_sma3nz_amean","eGeMAPS_F1amplitudeLogRelF0_sma3nz_stddevNorm",
        "eGeMAPS_F2frequency_sma3nz_amean","eGeMAPS_F2frequency_sma3nz_stddevNorm",
        "eGeMAPS_F2bandwidth_sma3nz_amean","eGeMAPS_F2bandwidth_sma3nz_stddevNorm",
        "eGeMAPS_F2amplitudeLogRelF0_sma3nz_amean","eGeMAPS_F2amplitudeLogRelF0_sma3nz_stddevNorm",
        "eGeMAPS_F3frequency_sma3nz_amean","eGeMAPS_F3frequency_sma3nz_stddevNorm",
        "eGeMAPS_F3bandwidth_sma3nz_amean","eGeMAPS_F3bandwidth_sma3nz_stddevNorm",
        "eGeMAPS_F3amplitudeLogRelF0_sma3nz_amean","eGeMAPS_F3amplitudeLogRelF0_sma3nz_stddevNorm",
        "eGeMAPS_alphaRatioV_sma3nz_amean","eGeMAPS_alphaRatioV_sma3nz_stddevNorm",
        "eGeMAPS_hammarbergIndexV_sma3nz_amean","eGeMAPS_hammarbergIndexV_sma3nz_stddevNorm",
        "eGeMAPS_slopeV0-500_sma3nz_amean","eGeMAPS_slopeV0-500_sma3nz_stddevNorm",
        "eGeMAPS_slopeV500-1500_sma3nz_amean","eGeMAPS_slopeV500-1500_sma3nz_stddevNorm",
        "eGeMAPS_spectralFluxV_sma3nz_amean","eGeMAPS_spectralFluxV_sma3nz_stddevNorm",
        "eGeMAPS_mfcc1V_sma3nz_amean","eGeMAPS_mfcc1V_sma3nz_stddevNorm",
        "eGeMAPS_mfcc2V_sma3nz_amean","eGeMAPS_mfcc2V_sma3nz_stddevNorm",
        "eGeMAPS_mfcc3V_sma3nz_amean","eGeMAPS_mfcc3V_sma3nz_stddevNorm",
        "eGeMAPS_mfcc4V_sma3nz_amean","eGeMAPS_mfcc4V_sma3nz_stddevNorm",
        "eGeMAPS_alphaRatioUV_sma3nz_amean","eGeMAPS_hammarbergIndexUV_sma3nz_amean",
        "eGeMAPS_slopeUV0-500_sma3nz_amean","eGeMAPS_slopeUV500-1500_sma3nz_amean",
        "eGeMAPS_spectralFluxUV_sma3nz_amean",
        "eGeMAPS_loudnessPeaksPerSec","eGeMAPS_VoicedSegmentsPerSec","eGeMAPS_MeanVoicedSegmentLengthSec",
        "eGeMAPS_StddevVoicedSegmentLengthSec","eGeMAPS_MeanUnvoicedSegmentLength",
        "eGeMAPS_StddevUnvoicedSegmentLength","eGeMAPS_equivalentSoundLevel_dBp"
    }

def get_all_language_features():
    return get_linguistic_features() | get_acoustic_features()