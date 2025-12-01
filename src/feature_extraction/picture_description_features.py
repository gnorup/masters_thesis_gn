# calculate features for combined picture description (cookieTheft + picnicScene), capped at 60s, 120s, & 300s

import os
import sys
import hashlib
import random
import tempfile
import pandas as pd
from pydub import AudioSegment

# add project root
sys.path.append("/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/src")

from config.constants import DATA_DIRECTORY, GIT_DIRECTORY, ID_COL

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

from feature_extraction.acoustic import (
    extract_acoustic_features as base_extract_acoustic_features,
    extract_egemaps as base_extract_egemaps,
)

from feature_extraction.helpers import (
    load_audio_file, load_audio_durations, load_transcriptions_df,
    infer_wav_duration_seconds, load_task_word_timestamps,
    append_row_atomic, save_checkpoint, load_checkpoint,
)

# config
PD_TASKS = ["cookieTheft", "picnicScene"]
OUT_DIR = os.path.join(GIT_DIRECTORY, "results", "features")
OUTPUTS = {
    "picture_description.csv": 300, # ≤ 5:00
    "picture_description_1min.csv": 60, # 1:00
    "picture_description_2min.csv": 120, # 2:00
}

# local helpers: handle in-memory audio segments for feature extraction on trimmed tasks
def extract_acoustic_features(*, audio_path=None, text=None, duration=None, audio_segment=None):
    if audio_segment is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), "tmp_picdesc_audio.wav")
        audio_segment.export(tmp_path, format="wav")
        return base_extract_acoustic_features(tmp_path, text, duration)
    return base_extract_acoustic_features(audio_path, text, duration)

def extract_egemaps(*, audio_path=None, audio_segment=None):
    if audio_segment is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), "tmp_picdesc_audio.wav")
        audio_segment.export(tmp_path, format="wav")
        return base_extract_egemaps(tmp_path)
    return base_extract_egemaps(audio_path)

# shuffle what picture description task is processed first
def task_order_for_subject(subject_id, base_tasks=None,seed=None):
    if base_tasks is None:
        base_tasks = PD_TASKS
    tasks = list(base_tasks)

    env_seed = os.environ.get("PICTURE_TASKS_SEED")
    if env_seed is not None and env_seed.strip():
        seed = int(env_seed)

    if seed is None:
        seed = int(hashlib.sha1(str(subject_id).encode()).hexdigest(), 16) % (10**8)

    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks

# concatenate picture description tasks (cookieTheft + picnicScene)
def combine_picture_description_streams(subject_id, task_order=None):
    subject_folder = os.path.join(DATA_DIRECTORY, subject_id)
    trans = load_transcriptions_df(subject_folder)

    if task_order is None:
        task_order = PD_TASKS

    combined_audio = AudioSegment.silent(duration=0)
    total_offset = 0.0
    tasks_included, words_blocks, text_fallbacks = [], [], []

    for task in task_order:
        wav = load_audio_file(subject_folder, task)
        if wav is None:
            continue

        seg = AudioSegment.from_wav(wav)
        dur_audio = len(seg) / 1000.0

        # CSV duration if present; else infer from wav
        dur_csv = load_audio_durations(subject_folder, task)
        if dur_csv is None:
            dur_csv = infer_wav_duration_seconds(wav)
        dur = max(dur_csv or 0.0, dur_audio)

        wdf = load_task_word_timestamps(subject_id, task)
        if wdf is not None:
            wdf2 = wdf.copy()
            wdf2["start"] = wdf2["start"].astype(float) + total_offset
            wdf2["end"]   = wdf2["end"].astype(float) + total_offset
            words_blocks.append(wdf2)
        else:
            # fallback to ASR text if available
            t = trans[trans["task"] == task]["text_google"].values
            if len(t) and isinstance(t[0], str) and t[0].strip():
                text_fallbacks.append(t[0].strip())

        combined_audio += seg
        total_offset += dur
        tasks_included.append(task)

    combined_words = pd.concat(words_blocks, ignore_index=True) if words_blocks else None
    if combined_words is not None and not combined_words.empty:
        combined_text = " ".join(combined_words["word"].tolist())
    else:
        combined_text = " ".join(text_fallbacks).strip()

    if not combined_text:
        return None, None, 0.0, [], None

    return combined_text, combined_audio, total_offset, tasks_included, combined_words

# return transcriptions and recordings for different durations
def truncate_by_seconds(combined_text, combined_audio, combined_words, cap_seconds):
    if cap_seconds is None:
        return combined_text, combined_audio, combined_words
    ms = int(cap_seconds * 1000)
    audio_cut = combined_audio[:ms]
    if combined_words is not None and not combined_words.empty:
        kept = combined_words[combined_words["end"] <= cap_seconds].copy()
        text_cut = " ".join(kept["word"].tolist())
        return text_cut, audio_cut, kept

    return combined_text, audio_cut, None

# calculate features for one time-capped slice
def compute_features_for_slice(subject_id, text, audio_cut, words_cut):

    aoa_lexicon = load_aoa_lexicon()
    frequency_lexicon = load_frequency_norms()
    concreteness_lexicon = load_concreteness_lexicon()

    used_dur = len(audio_cut) / 1000.0

    features = {
        ID_COL: subject_id,
        "duration_used_sec": used_dur,
        "n_words": n_words(text),
        "ttr": ttr(text),
        "avg_word_length": avg_word_length(text),
        "filler_word_ratio": filler_word_ratio(text),
        "brunets_index": brunets_index(text),
        "honores_statistic": honores_statistic(text),
        "guirauds_statistic": guirauds_statistic(text),
        "light_verb_ratio": light_verb_ratio(text),
        "empty_word_ratio": empty_word_ratio(text),
        "nid_ratio": nid_ratio(text),
        "adjacent_repetitions": adjacent_repetitions(text),
    }
    features.update(mattr(text, window_sizes=[10, 20, 30, 40, 50]))
    features.update(pos_ratios_spacy(text))
    features.update(calculate_fluency_features(text))

    for name, lex in [
        ("aoa",  aoa_lexicon),
        ("freq", frequency_lexicon),
        ("concr", concreteness_lexicon),
    ]:
        features[f"{name}_nouns"]   = compute_avg_by_pos(text, lex, ["NOUN"])
        features[f"{name}_verbs"]   = compute_avg_by_pos(text, lex, ["VERB"])
        features[f"{name}_content"] = compute_avg_by_pos(text, lex, ["NOUN", "VERB", "ADJ"])

    features.update(extract_acoustic_features(audio_segment=audio_cut, text=text, duration=used_dur))
    features.update(extract_egemaps(audio_segment=audio_cut))

    # article_pause_contentword on cut timestamps
    if words_cut is not None and not words_cut.empty:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpf:
            tmp_path = tmpf.name
        words_cut[["word", "start", "end"]].to_csv(tmp_path, index=False)
        try:
            apc = article_pause_contentword(tmp_path)
        except Exception as e:
            print(f"[WARNING] Failed to compute article_pause_contentword "
                  f"for subject {subject_id} (slice cap) from {tmp_path}: {e}")
            apc = None
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        features["article_pause_contentword"] = apc
    else:
        features["article_pause_contentword"] = None

    return pd.DataFrame([features])

# run feature extraction
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUT_DIR, "_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    subjects = sorted([
        s for s in os.listdir(DATA_DIRECTORY)
        if os.path.isdir(os.path.join(DATA_DIRECTORY, s)) and s.isdigit()
    ], key=lambda x: int(x))

    # load processed sets (from checkpoint or existing CSV)
    processed: dict[str, set[str]] = {}
    for fname in OUTPUTS.keys():
        ckpt_path = os.path.join(ckpt_dir, fname + ".json")
        done = load_checkpoint(ckpt_path)
        if not done:
            out_path = os.path.join(OUT_DIR, fname)
            if os.path.exists(out_path):
                try:
                    dfp = pd.read_csv(out_path, usecols=[ID_COL])
                    done = set(dfp[ID_COL].astype(str))
                except Exception:
                    done = set()
        processed[fname] = done

    for sid in subjects:
        subj_folder = os.path.join(DATA_DIRECTORY, sid)
        if not any(os.path.exists(os.path.join(subj_folder, f"{t}.wav")) for t in PD_TASKS):
            print(f"[skip] {sid}: no picture-description audio")
            continue

        order = task_order_for_subject(sid)

        combined = combine_picture_description_streams(sid, task_order=order)
        if combined[3] == []:
            print(f"[skip] {sid}: no picture-description data")
            continue
        combined_text, combined_audio, _total, tasks_included, combined_words = combined

        # compute once, reuse for each cap
        for fname, cap in OUTPUTS.items():
            out_path = os.path.join(OUT_DIR, fname)
            ckpt_path = os.path.join(ckpt_dir, fname + ".json")

            if sid in processed[fname]:
                print(f"[{fname}] already processed {sid}, skipping")
                continue

            print(f"[{fname}] processing {sid} (cap={cap}s)")
            text_cut, audio_cut, words_cut = truncate_by_seconds(
                combined_text, combined_audio, combined_words, cap
            )

            try:
                df_slice = compute_features_for_slice(
                    sid, text=text_cut, audio_cut=audio_cut, words_cut=words_cut
                )
                # columns for potential order effects
                df_slice["first_task"] = tasks_included[0] if tasks_included else None
                df_slice["task_order"] = "|".join(tasks_included) if tasks_included else None

                append_row_atomic(out_path, df_slice)
                processed[fname].add(sid)
                save_checkpoint(ckpt_path, processed[fname])
            except Exception as e:
                print(f"[{fname}] ERROR {sid}: {e}")
                continue

    print("\ndone. wrote/updated:")
    for fname in OUTPUTS.keys():
        print(" -", os.path.join(OUT_DIR, fname))

if __name__ == "__main__":
    main()
