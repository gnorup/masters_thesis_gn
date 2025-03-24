# DELETE THIS

import os  # for handling file paths and directories
import pandas as pd  # for handling CSV files efficiently

# Define base and output directories
base_dir = "/Users/gilanorup/Desktop/Studium/MSc/MA/dataset"
output_dir = "/Users/gilanorup/Desktop/Studium/MSc/MA/code/masters_thesis_gn/results/features"
os.makedirs(output_dir, exist_ok=True)

# Define the output file
output_file = os.path.join(output_dir, "cookieTheft.csv")

# Load data
def process_speech_fluency():
    input_file = os.path.join(base_dir, "combined_audio_durations.csv")
    df = pd.read_csv(input_file)

    # Calculate the average speaking time across spontaneous speech
    df["speech_fluency"] = df[["cookieTheft", "picnicScene", "journaling"]].mean(axis=1)

    # Check if the file already exists
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
    else:
        df_existing = pd.DataFrame()

    # Ensure Subject_ID is present
    df_speech_fluency = df[["Subject_ID", "speech_fluency"]]
    if "Subject_ID" not in df_existing:
        df_existing["Subject_ID"] = df_speech_fluency["Subject_ID"]

    # Merge new feature
    df_existing = df_existing.merge(df_speech_fluency, on="Subject_ID", how="outer")

    # Save updated file
    df_existing.to_csv(output_file, index=False)
    print(f"Updated {output_file} with speech_fluency")

if __name__ == "__main__":
    process_speech_fluency()
    print("Speech fluency analysis complete")
