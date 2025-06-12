# feature idea: Haider et al. (2020)
# includes Fundamental Frequency F0 (Shankar et al., 2025)

import opensmile

# set up the smile extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_egemaps(audio_path):
    df = smile.process_file(audio_path)

    # get the first (and only) row as a dictionary
    features = df.iloc[0].to_dict()

    # prefix keys with "eGeMAPS_"
    features = {f"eGeMAPS_{key}": value for key, value in features.items()}

    return features
