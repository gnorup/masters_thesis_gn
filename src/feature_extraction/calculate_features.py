import os
import pandas as pd
from features import (
    n_words, n_sentences, mwps, count_pos_spacy, pos_ratios_spacy,
    hesitation_ratio, speech_tempo, articulation_rate, shimmer, jitter,
    ttr, mattr, lexical_density, concreteness_score, aoa_average,
    speech_rate, speaking_time_ratio, pause_ratio
)

def process_features(task):
    """Load transcriptions and audio files, calculate features, and save results."""
    base_dir = "/Volumes/methlab_data/LanguageHealthyAging/data"
    output_dir = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/results/features"
    os.makedirs(output_dir, exist_ok=True)

    # load concreteness and AoA lexicons once
    from features.concreteness import load_concreteness_lexicon
    from features.aoa_average import load_aoa_lexicon
    concreteness_lexicon = load_concreteness_lexicon()
    aoa_lexicon = load_aoa_lexicon()

    output_file = os.path.join(output_dir, f"{task}.csv")
    feature_data = []

    for folder_number in range(41, 1371):
        subject_folder = os.path.join(base_dir, str(folder_number))

        # load duration of audio file
        total_duration = None
        duration_path = os.path.join(subject_folder, "audio_durations.csv") # path to audio duration file
        if os.path.exists(duration_path): # checks if file exists
            try:
                duration_df = pd.read_csv(duration_path) # tries to load csv
                match = duration_df[duration_df["task"] == task] # filters data frame to row for current task
                if not match.empty:
                    total_duration = match["duration"].values[0] # if row exists -> store duration value
            except Exception as e:
                print(f"⚠ error reading duration for subject {folder_number}: {e}")

        text_file_path = os.path.join(subject_folder, "ASR", "transcriptions.csv")
        audio_file_path = os.path.join(subject_folder, f"{task}.wav")

        # init variables
        text_features = {
            "n_words": None, "n_sentences": None, "mwps": None, "ttr": None, "mattr": None,
            "lexical_density": None, "concreteness_score": None, "aoa_average": None
        }
        pos_counts, pos_ratios = {}, {}
        fluency_features = {
            "hesitation_ratio": None, "speech_tempo": None, "speech_rate": None,
            "articulation_rate": None, "pause_ratio": None, "speaking_time_ratio": None
        }
        prosody_features = {"shimmer": None, "jitter": None}

        # process text
        if os.path.exists(text_file_path):
            df = pd.read_csv(text_file_path)
            text_row = df[df['task'] == task]['text_google']
            if not text_row.empty:
                text = text_row.iloc[0]
                if isinstance(text, float):
                    text = ""

                # linguistic features
                text_features["n_words"] = n_words(text)
                text_features["n_sentences"] = n_sentences(text)
                text_features["mwps"] = mwps(text)
                text_features["ttr"] = ttr(text)
                text_features["mattr"] = mattr(text)
                text_features["lexical_density"] = lexical_density(text)
                text_features["concreteness_score"] = concreteness_score(text, concreteness_lexicon)
                text_features["aoa_average"] = aoa_average(text, aoa_lexicon)

                pos_counts = count_pos_spacy(text)
                pos_ratios = pos_ratios_spacy(text)

                # fill in missing POS keys with None to ensure consistency
                all_pos_tags = [
                    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
                    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "OTHER"
                ]
                for tag in all_pos_tags:
                    pos_counts.setdefault(tag, None)
                    pos_ratios.setdefault(tag + "_ratio", pos_ratios.pop(tag, None))

                if total_duration:
                    fluency_features["speech_rate"] = speech_rate(text, total_duration)

        # process audio
        if os.path.exists(audio_file_path):
            fluency_features["speech_tempo"] = speech_tempo(audio_file_path)
            fluency_features["articulation_rate"] = articulation_rate(text, audio_path=audio_file_path)
            fluency_features["hesitation_ratio"] = hesitation_ratio(audio_file_path)
            prosody_features["shimmer"] = shimmer(audio_file_path)

            try:
                prosody_features["jitter"] = jitter(audio_file_path)
            except:
                prosody_features["jitter"] = None

            if total_duration:
                fluency_features["speaking_time_ratio"] = speaking_time_ratio(audio_file_path, total_duration)
                fluency_features["pause_ratio"] = pause_ratio(audio_file_path, total_duration)

        # only include subjects where at least one feature was extracted
        if any(value is not None for value in list(text_features.values()) + list(fluency_features.values()) + list(prosody_features.values())):
            row = [folder_number] + list(text_features.values()) + \
                  [pos_counts.get(k) for k in sorted(pos_counts.keys())] + \
                  [pos_ratios.get(k) for k in sorted(pos_ratios.keys())] + \
                  list(fluency_features.values()) + list(prosody_features.values())
            feature_data.append(row) # adds single subject's row to list feature_data

    # define column names (same order as row values)
    columns = ["Subject_ID"] + list(text_features.keys()) + \
              sorted(pos_counts.keys()) + sorted(pos_ratios.keys()) + \
              list(fluency_features.keys()) + list(prosody_features.keys())

    df_features = pd.DataFrame(feature_data, columns=columns) # create pandas data frame from feature_data

    # drop any rows that are fully empty besides Subject_ID
    df_features = df_features.dropna(how="all", subset=df_features.columns.difference(["Subject_ID"]))

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
