import sys
# add the root of the project (where src/ lives) to the path
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

import os
import pandas as pd

from feature_extraction.features import (
    n_words, clean_text, pos_ratios_spacy, filler_word_ratio,
    ttr, mattr, concreteness_score, aoa_average, average_word_length,
    brunets_index, honores_statistic, guirauds_statistic
)

from feature_extraction.audio import (
    speech_rate, speech_tempo, acoustic_features,
    extract_egemaps, VoiceActivityDetector
)

from feature_extraction.features.concreteness import load_concreteness_lexicon
from feature_extraction.features.aoa_average import load_aoa_lexicon

from config.constants import DATA_DIRECTORY
from config.constants import GIT_DIRECTORY



def load_audio_durations(subject_folder, task):
    """Returns duration for a given task if available, else None."""
    duration_path = os.path.join(subject_folder, "audio_durations.csv")  # path to audio duration file
    try:
        df = pd.read_csv(duration_path) # tries to load csv
        match = df[df["task"] == task] # filters data frame to row for current task
        return match["duration"].values[0] if not match.empty else None # if row exists -> store duration value
    except:
        return None

def load_transcription(subject_folder, task):
    """Returns transcription text for a given task if available, else empty string."""
    text_file_path = os.path.join(subject_folder, "ASR", "transcriptions.csv")
    try:
        df = pd.read_csv(text_file_path)
        match = df[df["task"] == task]["text_google"]
        return match.iloc[0] if not match.empty else ""
    except:
        return ""

def load_audio_file(subject_folder, task):
    """Returns path to audio file for a given task if available, else None."""
    path = os.path.join(subject_folder, f"{task}.wav")
    return path if os.path.exists(path) else None


def process_features(task):
    """Load transcriptions and audio files, calculate features, and save results."""
    base_dir = DATA_DIRECTORY
    output_dir = os.path.join(GIT_DIRECTORY, "results/features")
    os.makedirs(output_dir, exist_ok=True)

    # load concreteness and AoA lexicons once
    concreteness_lexicon = load_concreteness_lexicon()
    aoa_lexicon = load_aoa_lexicon()

    output_file = os.path.join(output_dir, f"{task}.csv")
    feature_data = []

    # get all valid subject folders once at the start
    valid_subjects = sorted([
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and name.isdigit()
    ], key=lambda x: int(x))  # convert to int before sorting

    # iterate over all valid subject folders in the base directory
    for subject_id in valid_subjects:
        subject_folder = os.path.join(base_dir, subject_id)

        print(f"Calculating features for subject {subject_id}...")

        total_duration = load_audio_durations(subject_folder, task)
        text = load_transcription(subject_folder, task)
        audio_file_path = load_audio_file(subject_folder, task)

        text_features = {
            "n_words": None, "ttr": None, "filler_word_ratio": None,
            "concreteness_score": None, "aoa_average": None
        }
        pos_ratios = {}
        acoustic = {}
        eGeMAPS = {}

        if text:
            # linguistic features
            text_features["n_words"] = n_words(text)
            text_features["ttr"] = ttr(text)
            text_features.update(mattr_all_windows(text, window_sizes=[10, 20, 30, 40, 50]))
            text_features["filler_word_ratio"] = filler_word_ratio(text)
            text_features["concreteness_score"] = concreteness_score(text, concreteness_lexicon)
            text_features["aoa_average"] = aoa_average(text, aoa_lexicon)
            text_features["average_word_length"] = average_word_length(text)
            text_features["brunets_index"] = brunets_index(text)
            text_features["honores_statistic"] = honores_statistic(text)
            text_features["guirauds_statistic"] = guirauds_statistic(text)

            pos_ratios = pos_ratios_spacy(text)

            if total_duration:
                acoustic["speech_rate"] = speech_rate(text, total_duration)

        if audio_file_path and total_duration:
            acoustic.update(acoustic_features(audio_file_path, text, total_duration))
            eGeMAPS = extract_egemaps(audio_file_path)

        # only include subjects where at least one feature was extracted
        if any(value is not None for value in
               list(text_features.values()) + list(pos_ratios.values()) + list(acoustic.values()) + list(
                       eGeMAPS.values())):
            row = {
                'Subject_ID': subject_id,
                **text_features,
                **pos_ratios,
                **acoustic,
                **eGeMAPS
            }
            feature_data.append(row) # adds single subject's row to list feature_data

    df_features = pd.DataFrame(feature_data) # create pandas data frame from feature_data

    # merge with existing file if present
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        cols_to_replace = [col for col in df_features.columns if col != "Subject_ID"]
        df_existing = df_existing.drop(columns=cols_to_replace, errors="ignore")
        df_features = df_existing.merge(df_features, on="Subject_ID", how="outer")

    df_features.to_csv(output_file, index=False) # saves final merged data frame to output csv
    print(f"updated {output_file} with extracted features")


if __name__ == "__main__":
    task_name = "cookieTheft"
    print(f"\nprocessing all subjects for task: {task_name}...\n")
    process_features(task_name)
    print("\nfeature extraction complete")
