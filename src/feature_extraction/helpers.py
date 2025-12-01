import os
import contextlib
import wave
import json
import pandas as pd
from pydub import AudioSegment

from config.constants import GIT_DIRECTORY, WORD_TIMESTAMPS_DATA, ID_COL

# load data for feature extraction
def load_transcription(subject_folder, task):
    """
    Return transcription for given task if available.
    """
    path = os.path.join(subject_folder, "ASR", "transcriptions.csv")
    try:
        df = pd.read_csv(path)
        transcription = df[df["task"] == task]["text_google"]
        return transcription.iloc[0] if not transcription.empty else ""
    except Exception as e:
        print(f"[WARNING] Failed to load transcription from {path} for task '{task}': {e}")
        return ""

def load_transcriptions_df(subject_folder: str) -> pd.DataFrame:
    """
    Load transcriptions for concatenation of picture description tasks.
    """
    p = os.path.join(subject_folder, "ASR", "transcriptions.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

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
    except Exception as e:
        print(f"[WARNING] Failed to read audio durations from {csv_path}: {e}")
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
            df[ID_COL] = df[ID_COL].astype(str)
            transcription = {(str(r[ID_COL]), str(r["task"])): r["transcription"]
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

def get_timestamp_csv(task, subject_id):
    """
    Access word timestamps to calculate article_pause_contentword.
    """
    timestamps_dir = WORD_TIMESTAMPS_DATA.format(task=task)
    path = os.path.join(timestamps_dir, f"{subject_id}.csv")
    return path if os.path.exists(path) else None

def load_task_word_timestamps(subject_id, task):
    """
    Load word timestamps for concatenation of picture description tasks.
    """
    p = get_timestamp_csv(task, subject_id)
    if not p:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    need = ["word", "start", "end"]
    if not all(c in df.columns for c in need):
        return None
    df = df[need].copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["word", "start", "end"])
    return df if not df.empty else None

# handle disruptions in feature extraction
def safe_upsert(existing, new_rows, key=ID_COL):
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

def append_row_atomic(out_path, row_df):
    """
    Append one row to csv and write header.
    """
    exists = os.path.exists(out_path)
    row_df.to_csv(out_path, mode="a", header=not exists, index=False)
    with open(out_path, "a") as f:
        f.flush()
        os.fsync(f.fileno())

def load_checkpoint(ckpt_path):
    """
    Load information on what subjects were already processed.
    """
    if not os.path.exists(ckpt_path):
        return set()
    try:
        with open(ckpt_path, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_checkpoint(ckpt_path, processed_ids):
    """
    Save which subjects have already been processed.
    """
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(processed_ids), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ckpt_path)