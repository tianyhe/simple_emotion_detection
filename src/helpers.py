import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pandas as pd
import json
import logging


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger"""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def plot_features(df, sample_index):
    sample_file = df.iloc[sample_index]

    # Load the audio file to get the waveform
    y, sr = librosa.load(sample_file["path"])

    plt.figure(figsize=(14, 16))

    # Waveform
    plt.subplot(4, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {sample_file['emotion']}")

    # MFCCs
    plt.subplot(4, 2, 2)
    mfccs = np.array(json.loads(sample_file["mfccs"]))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title("MFCC")

    # Spectral Centroid
    plt.subplot(4, 2, 3)
    spectral_centroid = np.array(json.loads(sample_file["spectral_centroid"]))
    frames = range(
        len(spectral_centroid[0])
    )  # Extracting the length from first element of the array
    t = librosa.frames_to_time(frames, sr=sr)
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(t, spectral_centroid.flatten(), color="r")
    plt.title("Spectral Centroid")

    # Zero Crossing Rate
    plt.subplot(4, 2, 4)
    zero_crossings = np.array(json.loads(sample_file["zero_crossing_rate"]))
    plt.plot(t, zero_crossings.flatten(), color="m")
    plt.title("Zero Crossing Rate")

    # RMS Energy
    plt.subplot(4, 2, 5)
    rms = np.array(json.loads(sample_file["rms"]))
    plt.plot(t, rms.flatten(), color="g")
    plt.title("RMS Energy")

    # Chroma Features
    plt.subplot(4, 2, 6)
    chroma = np.array(json.loads(sample_file["chroma"]))
    librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sr)
    plt.colorbar()
    plt.title("Chroma Features")

    # Mel Spectrogram
    plt.subplot(4, 2, 7)
    mel = np.array(json.loads(sample_file["mel"]))
    librosa.display.specshow(
        librosa.power_to_db(mel, ref=np.max), y_axis="mel", x_axis="time", sr=sr
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")

    # Spectral Contrast
    plt.subplot(4, 2, 8)
    contrast = np.array(json.loads(sample_file["contrast"]))
    librosa.display.specshow(contrast, x_axis="time", sr=sr)
    plt.colorbar()
    plt.title("Spectral Contrast")

    plt.tight_layout(pad=3.0)  # Adjust padding to prevent overlap

    # Tonnetz
    plt.figure(figsize=(14, 4))
    tonnetz = np.array(json.loads(sample_file["tonnetz"]))
    librosa.display.specshow(tonnetz, y_axis="tonnetz", x_axis="time", sr=sr)
    plt.colorbar()
    plt.title("Tonnetz")
    plt.tight_layout(pad=2.0)
    plt.show()


def plot_comparison(df, index1, index2):
    sample_file1 = df.iloc[index1]
    sample_file2 = df.iloc[index2]

    # Load the audio files to get the waveforms
    y1, sr1 = librosa.load(sample_file1["path"])
    y2, sr2 = librosa.load(sample_file2["path"])

    plt.figure(figsize=(14, 24))

    # Waveform
    plt.subplot(7, 1, 1)
    librosa.display.waveshow(
        y1, sr=sr1, alpha=0.5, label=sample_file1["emotion"], color="b"
    )
    librosa.display.waveshow(
        y2, sr=sr2, alpha=0.5, label=sample_file2["emotion"], color="r"
    )
    plt.title("Waveform")
    plt.legend()

    # Spectral Centroid
    plt.subplot(7, 1, 2)
    spectral_centroid1 = np.array(json.loads(sample_file1["spectral_centroid"]))
    spectral_centroid2 = np.array(json.loads(sample_file2["spectral_centroid"]))
    frames1 = range(len(spectral_centroid1[0]))
    frames2 = range(len(spectral_centroid2[0]))
    t1 = librosa.frames_to_time(frames1, sr=sr1)
    t2 = librosa.frames_to_time(frames2, sr=sr2)
    plt.plot(t1, spectral_centroid1.flatten(), color="b", label=sample_file1["emotion"])
    plt.plot(t2, spectral_centroid2.flatten(), color="r", label=sample_file2["emotion"])
    plt.title("Spectral Centroid")
    plt.legend()

    # Zero Crossing Rate
    plt.subplot(7, 1, 3)
    zero_crossings1 = np.array(json.loads(sample_file1["zero_crossing_rate"]))
    zero_crossings2 = np.array(json.loads(sample_file2["zero_crossing_rate"]))
    plt.plot(t1, zero_crossings1.flatten(), color="b", label=sample_file1["emotion"])
    plt.plot(t2, zero_crossings2.flatten(), color="r", label=sample_file2["emotion"])
    plt.title("Zero Crossing Rate")
    plt.legend()

    # RMS Energy
    plt.subplot(7, 1, 4)
    rms1 = np.array(json.loads(sample_file1["rms"]))
    rms2 = np.array(json.loads(sample_file2["rms"]))
    plt.plot(t1, rms1.flatten(), color="b", label=sample_file1["emotion"])
    plt.plot(t2, rms2.flatten(), color="r", label=sample_file2["emotion"])
    plt.title("RMS Energy")
    plt.legend()

    # MFCCs
    plt.subplot(7, 2, 9)
    mfccs1 = np.array(json.loads(sample_file1["mfccs"]))
    librosa.display.specshow(mfccs1, sr=sr1, x_axis="time")
    plt.colorbar()
    plt.title(f"MFCCs - {sample_file1['emotion']}")

    plt.subplot(7, 2, 10)
    mfccs2 = np.array(json.loads(sample_file2["mfccs"]))
    librosa.display.specshow(mfccs2, sr=sr2, x_axis="time")
    plt.colorbar()
    plt.title(f"MFCCs - {sample_file2['emotion']}")

    # Chroma Features
    plt.subplot(7, 2, 11)
    chroma1 = np.array(json.loads(sample_file1["chroma"]))
    librosa.display.specshow(chroma1, y_axis="chroma", x_axis="time", sr=sr1)
    plt.colorbar()
    plt.title(f"Chroma - {sample_file1['emotion']}")

    plt.subplot(7, 2, 12)
    chroma2 = np.array(json.loads(sample_file2["chroma"]))
    librosa.display.specshow(chroma2, y_axis="chroma", x_axis="time", sr=sr2)
    plt.colorbar()
    plt.title(f"Chroma - {sample_file2['emotion']}")

    plt.tight_layout(pad=3.0)  # Adjust padding to prevent overlap
    plt.show()
