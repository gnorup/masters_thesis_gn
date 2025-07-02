# cookieTheft_filtered1: first feature set, dropped features that were all zero upon inspection (PUNCT, SYM, OTHER) and eGeMAPS (except from shimmer & jitter)
# cookieTheft_filtered: second feature set without eGeMAPS (except from shimmer & jitter)

import pandas as pd
import os
import sys

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import GIT_DIRECTORY

# load features
task_name = "picnicScene"
features = pd.read_csv(os.path.join(GIT_DIRECTORY, f"results/features/cleaned/{task_name}_cleaned.csv"))
print(f"loaded {features.shape[0]} rows and {features.shape[1]} columns")

output_path = os.path.join(
    GIT_DIRECTORY,
    "results", "features", "filtered", f"{task_name}_filtered.csv"
)


# manual column filtering: add feature-names to drop
columns_to_drop = [
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_amean",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
    "eGeMAPS_loudness_sma3_amean",
    "eGeMAPS_loudness_sma3_stddevNorm",
    "eGeMAPS_loudness_sma3_percentile20.0",
    "eGeMAPS_loudness_sma3_percentile50.0",
    "eGeMAPS_loudness_sma3_percentile80.0",
    "eGeMAPS_loudness_sma3_pctlrange0-2",
    "eGeMAPS_loudness_sma3_meanRisingSlope",
    "eGeMAPS_loudness_sma3_stddevRisingSlope",
    "eGeMAPS_loudness_sma3_meanFallingSlope",
    "eGeMAPS_loudness_sma3_stddevFallingSlope",
    "eGeMAPS_spectralFlux_sma3_amean",
    "eGeMAPS_spectralFlux_sma3_stddevNorm",
    "eGeMAPS_mfcc1_sma3_amean",
    "eGeMAPS_mfcc1_sma3_stddevNorm",
    "eGeMAPS_mfcc2_sma3_amean",
    "eGeMAPS_mfcc2_sma3_stddevNorm",
    "eGeMAPS_mfcc3_sma3_amean",
    "eGeMAPS_mfcc3_sma3_stddevNorm",
    "eGeMAPS_mfcc4_sma3_amean",
    "eGeMAPS_mfcc4_sma3_stddevNorm",
    "eGeMAPS_jitterLocal_sma3nz_stddevNorm",
    "eGeMAPS_shimmerLocaldB_sma3nz_stddevNorm",
    "eGeMAPS_HNRdBACF_sma3nz_amean",
    "eGeMAPS_HNRdBACF_sma3nz_stddevNorm",
    "eGeMAPS_logRelF0-H1-H2_sma3nz_amean",
    "eGeMAPS_logRelF0-H1-H2_sma3nz_stddevNorm",
    "eGeMAPS_logRelF0-H1-A3_sma3nz_amean",
    "eGeMAPS_logRelF0-H1-A3_sma3nz_stddevNorm",
    "eGeMAPS_F1frequency_sma3nz_amean",
    "eGeMAPS_F1frequency_sma3nz_stddevNorm",
    "eGeMAPS_F1bandwidth_sma3nz_amean",
    "eGeMAPS_F1bandwidth_sma3nz_stddevNorm",
    "eGeMAPS_F1amplitudeLogRelF0_sma3nz_amean",
    "eGeMAPS_F1amplitudeLogRelF0_sma3nz_stddevNorm",
    "eGeMAPS_F2frequency_sma3nz_amean",
    "eGeMAPS_F2frequency_sma3nz_stddevNorm",
    "eGeMAPS_F2bandwidth_sma3nz_amean",
    "eGeMAPS_F2bandwidth_sma3nz_stddevNorm",
    "eGeMAPS_F2amplitudeLogRelF0_sma3nz_amean",
    "eGeMAPS_F2amplitudeLogRelF0_sma3nz_stddevNorm",
    "eGeMAPS_F3frequency_sma3nz_amean",
    "eGeMAPS_F3frequency_sma3nz_stddevNorm",
    "eGeMAPS_F3bandwidth_sma3nz_amean",
    "eGeMAPS_F3bandwidth_sma3nz_stddevNorm",
    "eGeMAPS_F3amplitudeLogRelF0_sma3nz_amean",
    "eGeMAPS_F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "eGeMAPS_alphaRatioV_sma3nz_amean",
    "eGeMAPS_alphaRatioV_sma3nz_stddevNorm",
    "eGeMAPS_hammarbergIndexV_sma3nz_amean",
    "eGeMAPS_hammarbergIndexV_sma3nz_stddevNorm",
    "eGeMAPS_slopeV0-500_sma3nz_amean",
    "eGeMAPS_slopeV0-500_sma3nz_stddevNorm",
    "eGeMAPS_slopeV500-1500_sma3nz_amean",
    "eGeMAPS_slopeV500-1500_sma3nz_stddevNorm",
    "eGeMAPS_spectralFluxV_sma3nz_amean",
    "eGeMAPS_spectralFluxV_sma3nz_stddevNorm",
    "eGeMAPS_mfcc1V_sma3nz_amean",
    "eGeMAPS_mfcc1V_sma3nz_stddevNorm",
    "eGeMAPS_mfcc2V_sma3nz_amean",
    "eGeMAPS_mfcc2V_sma3nz_stddevNorm",
    "eGeMAPS_mfcc3V_sma3nz_amean",
    "eGeMAPS_mfcc3V_sma3nz_stddevNorm",
    "eGeMAPS_mfcc4V_sma3nz_amean",
    "eGeMAPS_mfcc4V_sma3nz_stddevNorm",
    "eGeMAPS_alphaRatioUV_sma3nz_amean",
    "eGeMAPS_hammarbergIndexUV_sma3nz_amean",
    "eGeMAPS_slopeUV0-500_sma3nz_amean",
    "eGeMAPS_slopeUV500-1500_sma3nz_amean",
    "eGeMAPS_spectralFluxUV_sma3nz_amean",
    "eGeMAPS_loudnessPeaksPerSec",
    "eGeMAPS_VoicedSegmentsPerSec",
    "eGeMAPS_MeanVoicedSegmentLengthSec",
    "eGeMAPS_StddevVoicedSegmentLengthSec",
    "eGeMAPS_MeanUnvoicedSegmentLength",
    "eGeMAPS_StddevUnvoicedSegmentLength",
    "eGeMAPS_equivalentSoundLevel_dBp",
    "eGeMAPS_F0semitoneFrom27.5Hz_sma3nz_amean"
]

# drop specified columns
df_filtered = features.drop(columns=[col for col in columns_to_drop if col in features.columns])
print(f"dropped {len(columns_to_drop)} columns → new shape: {df_filtered.shape}")

# save filtered feature set
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_filtered.to_csv(output_path, index=False)
print(f"saved filtered features to: {output_path}")
