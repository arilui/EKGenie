# EKGenie

## Project Overview
This project consists of multiple components for analyzing, visualizing, and classifying EKG data. Below is a summary of the files included in this repository:

### Files

1. **DataVisualization.ipynb**
   - Visualizes EKG/ECG signals and class distributions.
   - Includes waveform plots, class-based heatmaps, and boxplots for feature exploration.

2. **FeatureEngineering.ipynb**
   - Feature engineering and cleaning on the MIT-BIH dataset.
   - Drops columns with mostly zero values, visualizes zero density, and prepares data for modeling.

3. **CNN.ipynb**
   - Deep learning notebook for classifying heartbeats using a 1D Convolutional Neural Network (CNN).
   - Loads processed data from FeatureEngineering, applies normalization, builds and trains a CNN, and evaluates with confusion matrix and ROC/AUC.

4. **RNN.ipynb**
   - Classifies heartbeats using a Recurrent Neural Network (RNN) architecture.
   - Similar workflow to CNN.ipynb but with RNN/LSTM layers.

5. **GRUN.ipynb** / **GRUN_withWeights.ipynb**
   - Implements Gated Recurrent Unit (GRU) models for heartbeat classification, with and without class weights.

6. **GrunCnn.ipynb**
   - Combines GRU and CNN layers for advanced sequence modeling.

7. **BidirectionalLSTM.ipynb**
   - Uses Bidirectional LSTM networks for improved sequence learning.

8. **Autoencoder.ipynb**
   - Explores unsupervised feature learning and anomaly detection with autoencoders.

9. **ARIMA.ipynb** / **SARIMA.ipynb**
   - Time series forecasting using ARIMA and SARIMA models.

10. **DynamicTimeWarping.ipynb**
    - Compares EKG signals using Dynamic Time Warping for similarity analysis.

11. **RandomForest.ipynb**
    - Implements a Random Forest classifier for heartbeat classification.

12. **naiveBayes.ipynb**
    - Implements a Naive Bayes classifier for heartbeat classification.
    - Compares performance with deep learning models.

13. **CleanData.ipynb**
    - Initial data cleaning and preprocessing.
    - Handles missing values, outlier removal, and basic exploratory analysis.

14. **ML_Test.ipynb**
    - Miscellaneous machine learning experiments and tests.

15. **HawraaJupyter.ipynb**
    - Data analysis and visualization of heartbeat datasets.
    - Tasks: loading data, PCA, correlation matrices, class distribution, and visualizations.

16. **Arduino.cpp**
    - Arduino code for reading EKG sensor data and transmitting it via serial communication.
    - Configured for a sampling rate of approximately 100Hz.

17. **Window.py**
    - Python GUI application built with Tkinter for recording and visualizing EKG data in real-time.
    - Features: serial port connection, data recording, live plotting, and CSV export.

## How to Use

### Jupyter Notebooks
- Open any notebook (e.g. `DataVisualization.ipynb`, `FeatureEngineering.ipynb`, `CNN.ipynb`, etc.) in Jupyter Notebook or VS Code.
- Follow the cells sequentially to load, preprocess, analyze, and model the data.

### Arduino Code
- Upload `Arduino.cpp` to an Arduino board connected to an EKG sensor.
- Ensure the correct analog pin and baud rate are configured.

### GUI Application
- Run `Window.py` using Python 3.
- Connect to the Arduino via the serial port and start recording EKG data.

## Requirements
- Python 3.x
- Libraries: `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn`, `tensorflow`, `serial`, `tkinter`, `kagglehub`
- Jupyter Notebook or VS Code for running notebooks
- Arduino IDE for uploading the Arduino code

## License
This project is licensed under the MIT License.
