import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fastdtw import fastdtw  # TODO: Replace with custom FastDTW implementation
from scipy.spatial.distance import euclidean
import json
import os
from datetime import datetime
from src.classical_dtw import classical_dtw_matrix, traceback


class EmotionDataset(Dataset):
    """
    Dataset class for emotion classification using temporal features.
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mfccs = np.array(json.loads(self.features[idx]["mfccs"]))
        mfccs = mfccs.T  # Transpose to (num_time_steps, 13)
        statistical_features = np.concatenate(
            [
                json.loads(self.features[idx][f"{feat}_mean"])
                + json.loads(self.features[idx][f"{feat}_std"])
                for feat in [
                    "mfccs",
                    "chroma",
                    "mel",
                    "contrast",
                    "tonnetz",
                    "spectral_centroid",
                    "zero_crossing_rate",
                    "rms",
                ]
            ]
        )
        return (
            torch.FloatTensor(mfccs),
            torch.FloatTensor(statistical_features),
            torch.LongTensor([self.labels[idx]]),
        )


class EmotionClassifier(nn.Module):
    """
    LSTM-based neural network for emotion classification using temporal features.
    """

    def __init__(self, mfcc_dim, stat_dim, hidden_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.lstm = nn.LSTM(mfcc_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + stat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, mfccs, stat_features):
        _, (hidden, _) = self.lstm(mfccs)
        combined = torch.cat((hidden.squeeze(0), stat_features), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=0.001,
    log_file=None,
    log_interval=10,
    save_interval=10,
):
    """
    Train the deep learning model for emotion classification.

    Parameters:
    - model: The neural network model to be trained.
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    - num_epochs (int): Number of epochs to train the model. Default is 50.
    - lr (float): Learning rate for the optimizer. Default is 0.001.
    - log_file: File object for logging the training progress. Default is None.
    - log_interval (int): Interval (in epochs) for logging training progress. Default is 10.
    - save_interval (int): Interval (in epochs) for saving the model. Default is 10.

    Returns:
    - results (list): List of dictionaries containing training and validation loss and accuracy for each epoch.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    results = []
    best_model_path = f"../models/best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for mfccs, stat_features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(mfccs, stat_features)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels.squeeze()).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for mfccs, stat_features, labels in val_loader:
                outputs = model(mfccs, stat_features)
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels.squeeze()).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            log_message = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            print(log_message)
            if log_file:
                log_file.write(log_message + "\n")

        results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            best_model_message = (
                f"New best model saved with validation accuracy: {val_acc:.2f}%"
            )
            print(best_model_message)
            if log_file:
                log_file.write(best_model_message + "\n")

    print(f"Best model saved as: {best_model_path}")
    if log_file:
        log_file.write(f"Best model saved as: {best_model_path}\n")

    return results


def align_sequences(sequences, fast_dtw=True):
    """
    Align sequences using either FastDTW or Classical DTW.

    Parameters:
    - sequences (list): List of sequences (numpy arrays) to be aligned.
    - fast_dtw (bool): If True, use FastDTW; otherwise, use Classical DTW.

    Returns:
    - aligned_sequences (numpy array): Aligned sequences.
    """
    max_len = max(seq.shape[1] for seq in sequences)
    reference = np.zeros((sequences[0].shape[0], max_len))

    aligned_sequences = []
    for seq in sequences:
        if fast_dtw:
            distance, path = fastdtw(reference.T, seq.T)
        else:
            cost_matrix = classical_dtw_matrix(reference.T, seq.T)
            path = traceback(cost_matrix)

        aligned_seq = np.zeros_like(reference)
        for ref_idx, seq_idx in path:
            aligned_seq[:, ref_idx] = seq[:, seq_idx]
        aligned_sequences.append(aligned_seq)

    return np.array(aligned_sequences)


def run_experiments(
    features_df,
    num_epochs=50,
    lr=0.001,
    hidden_size=64,
    batch_size=32,
    test_size=0.2,
    random_state=42,
    fast_dtw=True,
):
    """
    Run deep learning experiments for emotion classification using temporal features.

    Parameters:
    - features_df (DataFrame): DataFrame containing the features and labels for the dataset.
    - num_epochs (int): Number of epochs to train the model. Default is 50.
    - lr (float): Learning rate for the optimizer. Default is 0.001.
    - hidden_size (int): Size of the hidden layer in the model. Default is 64.
    - batch_size (int): Batch size for the DataLoader. Default is 32.
    - test_size (float): Proportion of the dataset to include in the validation split. Default is 0.2.
    - random_state (int): Random seed for reproducibility. Default is 42.
    - fast_dtw (bool): If True, use FastDTW; otherwise, use Classical DTW. Default is True.
    """

    # Create directories if they don't exist
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # Set up logging
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"../logs/dl_experiment_{current_time}_{'fast' if fast_dtw else 'classical'}_dtw.log"
    log_file = open(log_file_path, "w")

    # Load the data
    df = features_df

    # Prepare features and labels
    features = df.drop(["emotion"], axis=1).to_dict("records")
    labels = df["emotion"].values

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        features, encoded_labels, test_size=test_size, random_state=random_state
    )

    # Align MFCC sequences
    train_mfccs = [np.array(json.loads(sample["mfccs"])) for sample in X_train]
    val_mfccs = [np.array(json.loads(sample["mfccs"])) for sample in X_val]

    # Align sequences using FastDTW or classical DTW
    aligned_train_mfccs = align_sequences(train_mfccs, fast_dtw)
    aligned_val_mfccs = align_sequences(val_mfccs, fast_dtw)

    # Update the features with aligned MFCCs
    for i, sample in enumerate(X_train):
        sample["mfccs"] = json.dumps(aligned_train_mfccs[i].tolist())
    for i, sample in enumerate(X_val):
        sample["mfccs"] = json.dumps(aligned_val_mfccs[i].tolist())

    # Create datasets and dataloaders
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Calculate correct dimensions
    mfcc_dim = 13  # Number of MFCC coefficients
    stat_dim = (
        sum(
            len(json.loads(X_train[0][f"{feat}_mean"]))
            for feat in [
                "mfccs",
                "chroma",
                "mel",
                "contrast",
                "tonnetz",
                "spectral_centroid",
                "zero_crossing_rate",
                "rms",
            ]
        )
        * 2
    )  # * 2 for mean and std
    num_classes = len(le.classes_)

    log_file.write(f"DTW method: {'FastDTW' if fast_dtw else 'Classical DTW'}\n")
    log_file.write(f"MFCC dim: {mfcc_dim}\n")
    log_file.write(f"Statistical features dim: {stat_dim}\n")
    log_file.write(f"Hidden size: {hidden_size}\n")
    log_file.write(f"Number of classes: {num_classes}\n\n")

    model = EmotionClassifier(mfcc_dim, stat_dim, hidden_size, num_classes)

    # Train the model
    log_interval = 10
    save_interval = 10
    results = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        log_interval=log_interval,
        save_interval=save_interval,
        log_file=log_file,
    )

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = f"../results/dl_experiment_results_{current_time}_{'fast' if fast_dtw else 'classical'}_dtw.csv"
    results_df.to_csv(results_csv_path, index=False)

    log_file.write(f"\nResults saved to {results_csv_path}")
    log_file.close()

    # Print the best results
    best_result = results_df.loc[results_df["val_accuracy"].idxmax()]
    print(
        f"Best Validation Accuracy: {best_result['val_accuracy']:.4f} at Epoch {best_result['epoch']}"
    )
    print(f"Corresponding Training Accuracy: {best_result['train_accuracy']:.4f}")
    print(f"Experiment completed. Logs and results saved in ../results/ & ../logs")
    print(f"Model saved in ../models/")
