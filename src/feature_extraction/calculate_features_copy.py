import sys
import os
import contextlib
import wave
from asyncio import tasks

import pandas as pd
from pydub import AudioSegment

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import DATA_DIRECTORY, GIT_DIRECTORY

from feature_extraction.linguistic import (
    n_words, clean_text, tokenize, pos_ratios_spacy, filler_word_ratio,
    ttr, mattr, avg_word_length,
    light_verb_ratio, empty_word_ratio, nid_ratio, adjacent_repetitions,
    brunets_index, honores_statistic, guirauds_statistic
)
from feature_extraction.linguistic.psycholinguistic_features import (
    compute_avg_by_pos, load_aoa_lexicon, load_frequency_norms, load_concreteness_lexicon
)
from feature_extraction.linguistic.fluency_features import filled_pause_ratio, calculate_fluency_features
from feature_extraction.linguistic.article_pause_contentword import article_pause_contentword
from feature_extraction.acoustic import (
    count_phonemes, extract_acoustic_features,
    extract_egemaps, VoiceActivityDetector
)

TASKS = ["cookieTheft", "picnicScene", "journaling"]

# path to word timestamps
word_timestamps = "/Volumes/g_psyplafor_methlab$/Students/Gila/word_timestamps/{task}/google/timestamps"

def load_transcription(subject_folder, task):
    """
    Return transcription for given task if available.
    """
    path = os.path.join(subject_folder, "ASR", "transcriptions.csv")
    try:
        df = pd.read_csv(path)
        transcription = df[df["task"] == task]["text_google"]
        return transcription.iloc[0] if not transcription.empty else ""
    except Exception:
        return ""

def load_audio_file(subject_folder, task):
    """
    Return path to audio file for given task if available.
    """
    path = os.path.join(subject_folder, f"{task}.wav")
    return path if os.path.exists(path) else None

def infer_wav_duration_seconds(wav_path):
    """
    Infer duration of audio when missing duration from csv.
    """
    try:
        with contextlib.closing(wave.open(wav_path, 'r')) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception: # when file missing for example
        return None

def load_audio_durations(subject_folder, task):
    """
    Load audio duration from audio_durations.csv if available, else derive from .wav file.
    """
    csv_path = os.path.join(subject_folder, "audio_durations.csv")
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            audio = df[df["task"] == task]
            if not audio.empty:
                return float(audio["duration"].values[0])
    except Exception:
        pass
    # infer from wav file
    wav_path = load_audio_file(subject_folder, task)
    return infer_wav_duration_seconds(wav_path) if wav_path else None

def load_shortened_transcription():
    """
    For cases where audio > 5 minutes: load corresponding manually shortened transcription.
    """
    path = os.path.join(GIT_DIRECTORY, "data", "shortened_transcriptions.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["Subject_ID"] = df["Subject_ID"].astype(str)
            transcription = {(str(r["Subject_ID"]), str(r["task"])): r["transcription"]
                       for _, r in df.iterrows()}
            return transcription
        except Exception:
            return {}
    else:
        return {}

def trim_audio(subject_id, task, wav_path, max_seconds=300):
    """
    Trim audio files that exceed 5 minutes to 5 minutes.
    """
    out_dir = os.path.join(GIT_DIRECTORY, "data", "trimmed_audio") # store trimmed audio files for reuse
    os.makedirs(out_dir, exist_ok=True)
    trimmed_path = os.path.join(out_dir, f"{subject_id}_{task}_trimmed.wav")

    # if it already exists, reuse it
    if os.path.exists(trimmed_path):
        return trimmed_path

    # otherwise create it
    try:
        audio = AudioSegment.from_wav(wav_path)
        trimmed = audio[:max_seconds * 1000]  # ms
        trimmed.export(trimmed_path, format="wav")
        return trimmed_path
    except Exception:
        return wav_path # fallback to original file

def safe_upsert(existing, new_rows, key="Subject_ID"):
    """
    Safe upsert if feature extraction crashed.
    """
    if new_rows is None or new_rows.empty:
        return existing
    ex = existing.copy()
    nw = new_rows.copy()
    ex[key] = ex[key].astype(str)
    nw[key] = nw[key].astype(str)

    for c in nw.columns:
        if c not in ex.columns:
            ex[c] = pd.NA

    ex = ex.set_index(key)
    nw = nw.set_index(key)
    ex.update(nw)
    return ex.reset_index()

def article_pause_from_timestamps(task, subject_id):
    """
    Access word timestamps to calculate article_pause_contentword.
    """
    timestamps = word_timestamps.format(task=task)
    path = os.path.join(timestamps, f"{subject_id}.csv")
    if os.path.exists(path):
        try:
            return article_pause_contentword(path)
        except Exception:
            return None
    return None

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

        # enforce 5-minute rule for audio (and use manual shortened text if available)
        if total_duration and total_duration > 300:
            key = (str(subject_id), task)
            # use manually shortened transcription if available
            if key in shortened_map:
                text = shortened_map[key]

            # trim audio to 5 minutes and use trimmed file for acoustic features
            if audio_file_path:
                audio_file_path = trim_audio(subject_id, task, audio_file_path, max_seconds=300)

            # for feature calculations, treat duration as 300s
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
        apc = article_pause_from_timestamps(task, subject_id)
        if apc is not None:
            text_features["article_pause_contentword"] = apc

        if audio_file_path and total_duration:
            acoustic = extract_acoustic_features(audio_file_path, text, total_duration)
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
            feature_data.append(row)  # add single subject's row to list feature_data

    df_features = pd.DataFrame(feature_data)  # create pandas data frame from feature_data

    # merge with existing file if present (safe upsert)
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        if not df_existing.empty and "Subject_ID" in df_existing.columns:
            df_existing["Subject_ID"] = df_existing["Subject_ID"].astype(str)
        if not df_features.empty:
            df_features["Subject_ID"] = df_features["Subject_ID"].astype(str)
        df_features = safe_upsert(df_existing, df_features, key="Subject_ID")

    df_features.to_csv(output_file, index=False)  # save final merged data frame to output csv
    print(f"updated {output_file} with extracted features")


if __name__ == "__main__":
    for task_name in TASKS:
        print(f"\nprocessing all subjects for task: {task_name}\n")
        process_features(task_name)
        print("\nfeature extraction complete")