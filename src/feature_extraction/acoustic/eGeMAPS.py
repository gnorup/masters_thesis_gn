from opensmile import Smile, FeatureSet, FeatureLevel

# set up openSMILE extractor
smile = Smile(
    feature_set=FeatureSet.eGeMAPSv02,
    feature_level=FeatureLevel.Functionals
)

def extract_egemaps(audio_path):
    """
    Extract eGeMAPS features from an audio file using openSMILE based on Eyben et al. (2015).
    """
    df = smile.process_file(audio_path)
    features = df.iloc[0].to_dict() # convert to dictionary
    features = {f"eGeMAPS_{key}": value for key, value in features.items()}
    return features