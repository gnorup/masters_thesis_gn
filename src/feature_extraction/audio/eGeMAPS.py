# feature idea: Haider et al. (2020)

import opensmile

# set up the smile extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_egemaps(audio_path):
    """Extracts eGeMAPS features as a dictionary with names and values."""
    df = smile.process_file(audio_path)

    # get the first (and only) row as a dictionary
    features = df.iloc[0].to_dict()

    # prefix keys with "eGeMAPS_"
    features = {f"eGeMAPS_{key}": value for key, value in features.items()}

    return features
