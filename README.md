# Deep Fake Audio Detection

## Overview
This project implements a machine learning pipeline to detect deep fake audio using audio feature extraction and an XGBoost classifier. It processes audio files (`.wav` and `.mp3`) from the FakeOrReal dataset, extracts features like MFCC, spectral contrast, and zero-crossing rate, and trains a model to classify audio as real or fake. The pipeline includes data augmentation, handling of imbalanced datasets using SMOTE, and hyperparameter tuning for optimal performance.

## Features
- **Audio Feature Extraction**: Extracts Mel-frequency cepstral coefficients (MFCC), spectral contrast, and zero-crossing rate using `librosa`.
- **Data Augmentation**: Applies Gaussian noise, pitch shifting, and time stretching to enhance model robustness.
- **Class Imbalance Handling**: Uses SMOTE and class weighting to address imbalanced real and fake audio samples.
- **Model Training**: Employs an XGBoost classifier with optional hyperparameter tuning via grid search.
- **Evaluation**: Provides detailed metrics including accuracy, precision, recall, F1-score, and ROC AUC, along with a confusion matrix.
- **Prediction**: Supports real-time prediction on new audio files.

## Requirements
- Python 3.10+
- Libraries: `librosa`, `numpy`, `xgboost`, `imblearn`, `pydub`, `scikit-learn`
- Dataset: [FakeOrReal dataset](https://www.kaggle.com/datasets/jordankrist/the-fake-or-real-dataset) (or your own dataset with a similar structure)
- FFmpeg (for `.mp3` file processing)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/fares-mady/deep-fake-audio-detection.git
   cd deep-fake-audio-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install the dependencies:
   ```bash
   pip install librosa xgboost imblearn pydub scikit-learn numpy
   ```
3. Install FFmpeg for `.mp3` file support:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download and install from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage
1. **Prepare the Dataset**:
   - Place the FakeOrReal dataset in a directory (e.g., `data/`).
   - The dataset should have the structure:
     ```
     data/
     ├── for-2sec/
     │   ├── training/
     │   │   ├── real/
     │   │   └── fake/
     │   ├── testing/
     │   │   ├── real/
     │   │   └── fake/
     │   └── validation/
     ├── for-norm/
     ├── for-original/
     └── for-rerec/
     ```

2. **Run the Notebook**:
   - Open `deep-fake-audio-detection.ipynb` in Jupyter Notebook or JupyterLab.
   - Update the `for_base_path` variable to point to your dataset directory.
   - Execute the cells to install dependencies, load the dataset, train the model, and evaluate or make predictions.

3. **Make Predictions**:
   - After training, use the `predict_fake_audio` function to classify a new audio file:
     ```python
     result = predict_fake_audio(model, scaler, "path/to/your/audio.wav")
     print(f"Prediction: {result}")
     ```

## Dataset
The project uses the [FakeOrReal dataset](https://www.kaggle.com/datasets/jordankrist/the-fake-or-real-dataset) from Kaggle, which contains real and fake audio samples across multiple splits (`for-2sec`, `for-norm`, `for-original`, `for-rerec`). Each split includes `training`, `testing`, and `validation` folders with `real` and `fake` subdirectories.

## Model Performance
The trained XGBoost model achieves high performance, with metrics from the notebook:
- **Test Accuracy**: ~95.8%
- **F1-Score (macro)**: ~95.8%
- **ROC AUC**: ~99.1%
Detailed results, including precision, recall, and confusion matrix, are printed during evaluation.

## Limitations
- Requires FFmpeg for `.mp3` file processing.
- Some `.mp3` files may fail to decode due to format issues (as seen in the notebook output).
- Performance depends on the quality and diversity of the training dataset.

## Future Improvements
- Add support for additional audio features (e.g., chroma features, spectral rolloff).
- Explore deep learning models like CNNs or RNNs for improved feature extraction.
- Implement real-time audio streaming for live detection.
- Enhance error handling for corrupted audio files.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Submit a pull request with a clear description of changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Librosa](https://librosa.org/) for audio feature extraction.
- [XGBoost](https://xgboost.readthedocs.io/) for efficient classification.
- [Kaggle](https://www.kaggle.com/) for providing the FakeOrReal dataset.
