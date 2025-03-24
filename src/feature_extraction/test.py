import os
import librosa
import soundfile as sf
import pandas as pd
from features import (
    n_words, n_sentences, mwps, count_pos_spacy, pos_ratios_spacy,
    hesitation_ratio, speech_tempo, articulation_rate, shimmer, jitter,
    ttr, mattr, lexical_density, concreteness_score, aoa_average,
    speech_rate, speaking_time_ratio, pause_ratio
)
from features.concreteness import load_concreteness_lexicon
from features.aoa_average import load_aoa_lexicon

# Load lexicons once
concreteness_lexicon = load_concreteness_lexicon()
aoa_lexicon = load_aoa_lexicon()

TEST_SUBJECTS = [41, 43]
TASK = "cookieTheft"
BASE_DIR = "/Volumes/methlab_data/LanguageHealthyAging/data"

def is_valid_audio(audio_path):
    try:
        with sf.SoundFile(audio_path) as f:
            if f.samplerate not in [8000, 16000, 32000, 44100]:
                print(f"⚠️ {audio_path}: Unsupported sample rate ({f.samplerate} Hz). Converting to 16kHz...")
                return False
            if f.subtype not in ["PCM_16"]:
                print(f"⚠️ {audio_path}: Not 16-bit PCM WAV. Converting...")
                return False
        return True
    except Exception as e:
        print(f"❌ Invalid audio file: {audio_path} - {e}")
        return False

def convert_audio(audio_path):
    import tempfile
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y = librosa.util.normalize(y)
    temp_dir = tempfile.gettempdir()
    new_path = os.path.join(temp_dir, os.path.basename(audio_path).replace(".wav", "_converted.wav"))
    sf.write(new_path, y, 16000, subtype="PCM_16")
    print(f"✅ Converted {audio_path} → {new_path} (Mono, 16-bit PCM, 16kHz)")
    return new_path

def test_feature_extraction(subject_id, task):
    subject_folder = os.path.join(BASE_DIR, str(subject_id))
    text_file_path = os.path.join(subject_folder, "ASR", "transcriptions.csv")
    audio_file_path = os.path.join(subject_folder, f"{task}.wav")
    duration_path = os.path.join(subject_folder, "audio_durations.csv")

    total_duration = None
    if os.path.exists(duration_path):
        try:
            duration_df = pd.read_csv(duration_path)
            match = duration_df[duration_df["task"] == task]
            if not match.empty:
                total_duration = match["duration"].values[0]
        except Exception as e:
            print(f"⚠️ Error reading duration for subject {subject_id}: {e}")

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

    if os.path.exists(text_file_path):
        df = pd.read_csv(text_file_path)
        text_row = df[df['task'] == task]['text_google']
        if not text_row.empty:
            text = text_row.iloc[0]
            if isinstance(text, float):
                text = ""
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
            if total_duration:
                fluency_features["speech_rate"] = speech_rate(text, total_duration)
        else:
            print(f"⚠️ Subject {subject_id}: No text found for task '{task}'.")
    else:
        print(f"❌ Subject {subject_id}: Transcription file not found.")

    if os.path.exists(audio_file_path):
        use_audio_file = audio_file_path
        if not is_valid_audio(audio_file_path):
            use_audio_file = convert_audio(audio_file_path)
        try:
            fluency_features["speech_tempo"] = speech_tempo(audio_file_path)
            fluency_features["articulation_rate"] = articulation_rate(text, use_audio_file)
            fluency_features["hesitation_ratio"] = hesitation_ratio(use_audio_file)
            prosody_features["shimmer"] = shimmer(use_audio_file)
            prosody_features["jitter"] = jitter(use_audio_file)
            if total_duration:
                fluency_features["speaking_time_ratio"] = speaking_time_ratio(use_audio_file, total_duration)
                fluency_features["pause_ratio"] = pause_ratio(use_audio_file, total_duration)
        except Exception as e:
            print(f"⚠️ Error processing audio for Subject {subject_id}: {e}")
    else:
        print(f"⚠️ Subject {subject_id}: Audio file not found for '{task}'.")

    print(f"\n📊 **Results for Subject {subject_id}:**")
    print(text_features)
    print(pos_counts)
    print(pos_ratios)
    print(fluency_features)
    print(prosody_features)
    print("=" * 40)

if __name__ == "__main__":
    print(f"\n🔍 Running Feature Extraction for Task: {TASK}...\n")
    for subject_id in TEST_SUBJECTS:
        test_feature_extraction(subject_id, TASK)
    print("\n✅ Feature extraction tests completed!")
