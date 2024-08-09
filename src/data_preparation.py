import os
import json
import pandas as pd
import librosa
import numpy as np


def get_emotion(filename):
    """Extract emotion from filename and map to human-readable label."""
    emotion_code = filename.split("-")[2]
    emotion_map = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}
    return emotion_map.get(emotion_code, "unknown")


def filter_ravdess_files(base_dir):
    """Filter RAVDESS files based on project criteria."""
    relevant_files = []
    relevant_emotions = {"01", "03", "04", "05"}  # neutral, happy, sad, angry

    for actor_folder in os.listdir(base_dir):
        actor_path = os.path.join(base_dir, actor_folder)
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith(".wav"):
                    parts = filename.split("-")
                    if (
                        parts[0] == "03"  # audio-only
                        and parts[1] == "01"  # speech
                        and parts[2] in relevant_emotions
                    ):
                        full_path = os.path.join(actor_path, filename)
                        emotion = get_emotion(filename)
                        relevant_files.append(
                            {
                                "filename": filename,
                                "path": full_path,
                                "emotion": emotion,
                                "actor": actor_folder,
                            }
                        )

    return pd.DataFrame(relevant_files)


def extract_features(y, sr, n=13):
    # Extract MFCCs (n=13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n)

    # Extract other features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Compute mean and standard deviation of the features
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)
    spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
    spectral_centroid_std = np.std(spectral_centroid, axis=1)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate, axis=1)
    zero_crossing_rate_std = np.std(zero_crossing_rate, axis=1)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)

    # Combine all features into a single feature vector including 13 MFCCs
    combined_features = np.hstack(
        [
            mfccs.flatten(),
            mfccs_mean,
            mfccs_std,
            chroma.flatten(),
            chroma_mean,
            chroma_std,
            mel.flatten(),
            mel_mean,
            mel_std,
            contrast.flatten(),
            contrast_mean,
            contrast_std,
            tonnetz.flatten(),
            tonnetz_mean,
            tonnetz_std,
            spectral_centroid.flatten(),
            spectral_centroid_mean,
            spectral_centroid_std,
            zero_crossing_rate.flatten(),
            zero_crossing_rate_mean,
            zero_crossing_rate_std,
            rms.flatten(),
            rms_mean,
            rms_std,
        ]
    )

    # Return a dictionary of all features separately and the combined feature vector
    return {
        "mfccs": mfccs,
        "mfccs_mean": mfccs_mean,
        "mfccs_std": mfccs_std,
        "chroma": chroma,
        "chroma_mean": chroma_mean,
        "chroma_std": chroma_std,
        "mel": mel,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "contrast": contrast,
        "contrast_mean": contrast_mean,
        "contrast_std": contrast_std,
        "tonnetz": tonnetz,
        "tonnetz_mean": tonnetz_mean,
        "tonnetz_std": tonnetz_std,
        "spectral_centroid": spectral_centroid,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
        "zero_crossing_rate": zero_crossing_rate,
        "zero_crossing_rate_mean": zero_crossing_rate_mean,
        "zero_crossing_rate_std": zero_crossing_rate_std,
        "rms": rms,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "combined_features": combined_features,
    }


def prepare_data(base_dir, output_file):
    """Filter files and extract features."""
    df = filter_ravdess_files(base_dir)

    # Initialize columns for each feature
    feature_columns = [
        "mfccs",
        "mfccs_mean",
        "mfccs_std",
        "chroma",
        "chroma_mean",
        "chroma_std",
        "mel",
        "mel_mean",
        "mel_std",
        "contrast",
        "contrast_mean",
        "contrast_std",
        "tonnetz",
        "tonnetz_mean",
        "tonnetz_std",
        "spectral_centroid",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "zero_crossing_rate",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "rms",
        "rms_mean",
        "rms_std",
        "combined_features",
    ]

    for col in feature_columns:
        df[col] = None

    # Extract features and create separate columns for each feature
    for index, row in df.iterrows():
        y, sr = librosa.load(row["path"])
        feature_vector = extract_features(y, sr)

        # Loop through the features dictionary and save each feature in its own column
        for key, value in feature_vector.items():
            if isinstance(value, np.ndarray):
                # Save the numpy array as a JSON string to preserve the temporal structure
                value = json.dumps(value.tolist())
            df.at[index, key] = value

    # Save to CSV
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    base_directory = "./data/audio_speech_actors_01-24"
    output_file = "./data/ravdess_features.csv"
    df = prepare_data(base_directory, output_file)
    print(f"Data preparation complete. Output saved to {output_file}")
    print(f"Total files processed: {len(df)}")
    print("\nFiles per emotion:")
    print(df["emotion"].value_counts())
