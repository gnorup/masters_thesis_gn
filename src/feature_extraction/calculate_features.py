import os
import pandas as pd

from config.constants import DATA_DIRECTORY, GIT_DIRECTORY, ID_COL, TASKS

from feature_extraction.linguistic import (
    n_words, pos_ratios_spacy, filler_word_ratio, ttr, mattr, avg_word_length,
    light_verb_ratio, empty_word_ratio, nid_ratio, adjacent_repetitions,
    brunets_index, honores_statistic, guirauds_statistic
)
from feature_extraction.linguistic.psycholinguistic_features import (
    compute_avg_by_pos, load_aoa_lexicon, load_frequency_norms, load_concreteness_lexicon
)
from feature_extraction.linguistic.fluency_features import calculate_fluency_features
from feature_extraction.linguistic.article_pause_contentword import article_pause_contentword
from feature_extraction.acoustic import extract_acoustic_features, extract_egemaps

from feature_extraction.helpers import (
    load_transcription, load_audio_file, load_audio_durations,
    load_shortened_transcription, trim_audio, safe_upsert, get_timestamp_csv
)

# main feature extraction
def process_features(task):
    """
    Extract all linguistic and acoustic features for all subjects for a given task.
    """
    base_dir = DATA_DIRECTORY
    output_dir = os.path.join(GIT_DIRECTORY, "results/features")
    os.makedirs(output_dir, exist_ok=True)

    # load lexicons for psycholinguistic features
    concreteness_lexicon = load_concreteness_lexicon()
    aoa_lexicon = load_aoa_lexicon()
    frequency_lexicon = load_frequency_norms()

    # manually shortened transcriptions (only for >5min speakers, if present)
    shortened_map = load_shortened_transcription()

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

        print(f"calculating features for subject {subject_id}")

        total_duration = load_audio_durations(subject_folder, task)
        text = load_transcription(subject_folder, task)
        audio_file_path = load_audio_file(subject_folder, task)

        # 5-minute rule for audio
        if total_duration and total_duration > 300:
            key = (str(subject_id), task)
            # use manually shortened transcription if available
            if key in shortened_map:
                text = shortened_map[key]
            # trim audio to 5 minutes and use trimmed file for acoustic features
            if audio_file_path:
                audio_file_path = trim_audio(subject_id, task, audio_file_path, max_seconds=300)

            total_duration = 300.0

        text_features = {}
        pos_ratios = {}
        acoustic = {}
        eGeMAPS = {}

        if text:
            # linguistic features
            text_features["n_words"] = n_words(text)
            text_features["ttr"] = ttr(text)
            text_features.update(mattr(text, window_sizes=[10, 20, 30, 40, 50]))
            text_features["filler_word_ratio"] = filler_word_ratio(text)
            text_features["avg_word_length"] = avg_word_length(text)
            text_features["brunets_index"] = brunets_index(text)
            text_features["honores_statistic"] = honores_statistic(text)
            text_features["guirauds_statistic"] = guirauds_statistic(text)
            text_features["light_verb_ratio"] = light_verb_ratio(text)
            text_features["empty_word_ratio"] = empty_word_ratio(text)
            text_features["nid_ratio"] = nid_ratio(text)
            text_features["adjacent_repetitions"] = adjacent_repetitions(text)

            # AoA
            text_features["aoa_content"] = compute_avg_by_pos(text, aoa_lexicon, pos_tags=["NOUN", "VERB", "ADJ"])
            text_features["aoa_nouns"] = compute_avg_by_pos(text, aoa_lexicon, pos_tags=["NOUN"])
            text_features["aoa_verbs"] = compute_avg_by_pos(text, aoa_lexicon, pos_tags=["VERB"])

            # frequency
            text_features["freq_content"] = compute_avg_by_pos(text, frequency_lexicon, pos_tags=["NOUN", "VERB", "ADJ"])
            text_features["freq_nouns"] = compute_avg_by_pos(text, frequency_lexicon, pos_tags=["NOUN"])
            text_features["freq_verbs"] = compute_avg_by_pos(text, frequency_lexicon, pos_tags=["VERB"])

            # concreteness
            text_features["concr_content"] = compute_avg_by_pos(text, concreteness_lexicon, pos_tags=["NOUN", "VERB", "ADJ"])
            text_features["concr_nouns"] = compute_avg_by_pos(text, concreteness_lexicon, pos_tags=["NOUN"])
            text_features["concr_verbs"] = compute_avg_by_pos(text, concreteness_lexicon, pos_tags=["VERB"])

            pos_ratios = pos_ratios_spacy(text)

            fluency_features = calculate_fluency_features(text)
            text_features.update(fluency_features)

        # article_pause_contentword from timestamps
        timestamp_path = get_timestamp_csv(task, subject_id)
        if timestamp_path:
            try:
                apc = article_pause_contentword(timestamp_path)
                if apc is not None:
                    text_features["article_pause_contentword"] = apc
            except Exception as e:
                pass
                print(f"[WARNING] Failed to compute article_pause_contentword "
                      f"for subject {subject_id} in task '{task}': {e}")

        if audio_file_path and total_duration:
            acoustic = extract_acoustic_features(audio_file_path, text, total_duration)
            eGeMAPS = extract_egemaps(audio_file_path)

        # only include subjects where at least one feature was extracted
        if any(value is not None for value in
               list(text_features.values()) + list(pos_ratios.values()) + list(acoustic.values()) + list(
                       eGeMAPS.values())):
            row = {
                ID_COL: subject_id,
                **text_features,
                **pos_ratios,
                **acoustic,
                **eGeMAPS
            }
            feature_data.append(row)  # add single subject's row to list feature_data

    df_features = pd.DataFrame(feature_data)

    # merge with existing file if present
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        if not df_existing.empty and ID_COL in df_existing.columns:
            df_existing[ID_COL] = df_existing[ID_COL].astype(str)
        if not df_features.empty:
            df_features[ID_COL] = df_features[ID_COL].astype(str)
        df_features = safe_upsert(df_existing, df_features, key=ID_COL)

    # save features to csv
    df_features.to_csv(output_file, index=False)
    print(f"updated {output_file} with extracted features")


if __name__ == "__main__":
    for task_name in TASKS:
        print(f"\nprocessing all subjects for task: {task_name}\n")
        process_features(task_name)
        print("\nfeature extraction complete")