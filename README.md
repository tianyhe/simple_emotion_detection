# Simple Emotion Detection using Audio Features and Dynamic Time Warping (DTW)

## Overview
This project focuses on emotion detection using audio features and Dynamic Time Warping (DTW) distance metrics. It involves comparing the efficiency of classical DTW and FastDTW in terms of speed, accuracy, and scalability. The project includes data preparation, model development, and performance analysis.

## Project Structure
The project is organized as follows:

```
mood-detection/
├── data/
│   └── audio_speech_dataset/
├── log/
├── models/
├── results/
├── src/
│   ├── classical_dtw.py
│   ├── data_preparation.py
│   ├── dl_model.py
│   ├── fast_dtw.py
│   └── helpers.py
└── notes.ipynb
```

### Directories and Files
- **data/**: Contains the raw audio speech dataset and the saved extracted features files.
  - `audio_speech_dataset/`: Directory containing the raw audio data.
- **log/**: Contains log files from various experiments and runs.
- **models/**: Directory where the best models from the training are saved.
- **results/**: Directory containing the results of the experiments.
- **src/**: Contains all the source code for the project.
  - `classical_dtw.py`: Implementation of the classical DTW algorithm.
  - `data_preparation.py`: Scripts for preparing and preprocessing the audio data.
  - `dl_model.py`: Contains the deep learning model for emotion detection.
  - `fast_dtw.py`: Implementation of the FastDTW algorithm.
  - `helpers.py`: Helper functions used across various scripts.
- **notes.ipynb**: Jupyter notebook with detailed notes, visualizations, and analysis of the project.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```sh
   cd mood-detection
   ```
3. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Data Preparation
Prepare the audio data using the `data_preparation.py` script:
```sh
python src/data_preparation.py
```
This script will preprocess the audio data and extract the necessary features for emotion detection.

### Classical DTW
To run the classical DTW algorithm, use the `classical_dtw.py` script:
```sh
python src/classical_dtw.py
```
This script computes the DTW distance using the classical DTW algorithm.

### FastDTW
To run the FastDTW algorithm, use the `fast_dtw.py` script:
```sh
python src/fast_dtw.py
```
This script computes the DTW distance using the FastDTW algorithm.

### Deep Learning Model
Train and evaluate the deep learning model using the `dl_model.py` script:
```sh
python src/dl_model.py
```
This script trains a deep learning model for emotion detection and evaluates its performance.

## Notes and Analysis
Detailed notes, visualizations, and analysis can be found in the `notes.ipynb` Jupyter notebook. This notebook includes:
- Overview of the project
- Data exploration and visualization
- Comparison of classical DTW and FastDTW
- Model performance analysis

## Reflection
The project compares the classical DTW and FastDTW algorithms, focusing on:
- **Speed**: FastDTW is significantly faster than classical DTW, especially with larger datasets.
- **Accuracy**: Both algorithms show comparable accuracy, with minor differences depending on the dataset and parameters used.
- **Scalability**: FastDTW scales better with larger datasets, making it more suitable for real-time applications.

## Conclusion
This project demonstrates the effectiveness of DTW and its variants in emotion detection using audio features. It provides a comprehensive analysis of the trade-offs between classical DTW and FastDTW, offering insights into their performance in different scenarios.

## Acknowledgments
Special thanks to the contributors and the resources used throughout the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to reach out if you have any questions or need further assistance!