# EKGenie

## Project Overview
This project provides a comprehensive toolkit for EKG/ECG data analysis, visualization, and classification. It includes classic machine learning, deep learning, time series, and signal processing approaches, as well as tools for data collection and real-time visualization.

### Files

1. **Arduino.cpp**
   - Arduino code for reading EKG sensor data and transmitting via serial communication (approx. 100Hz sampling).
2. **ARIMA.ipynb**
   - Time series forecasting using ARIMA models.
3. **Autoencoder.ipynb**
   - Unsupervised feature learning and anomaly detection with autoencoders.
4. **AutoencoderWithRF.ipynb**
   - Combines autoencoder feature extraction with Random Forest classification.
5. **BidirectionalLSTM.ipynb**
   - Bidirectional LSTM for improved sequence learning on EKG data.
6. **CleanData.ipynb**
   - Initial data cleaning, missing value handling, and outlier removal.
7. **CleanDataUnweighted.ipynb**
   - Data cleaning and preprocessing without class weighting.
8. **CNN.ipynb**
   - 1D Convolutional Neural Network for heartbeat classification, with normalization, training, confusion matrix, and ROC/AUC analysis.
9. **CNNUnweightedNoFocalLoss.ipynb**
   - CNN model without class weighting or focal loss.
10. **cnn_attention.ipynb**
    - CNN model with attention mechanism for improved heartbeat classification.
11. **DataVisualization.ipynb**
    - Visualizes EKG/ECG signals and class distributions with waveform plots, heatmaps, and boxplots.
12. **DynamicTimeWarping.ipynb**
    - Signal similarity analysis using Dynamic Time Warping.
13. **FeatureEngineering.ipynb**
    - Cleans and engineers features, drops low-information columns, and prepares data for modeling.
14. **GRUN.ipynb**
    - Gated Recurrent Unit (GRU) model for heartbeat classification.
15. **GRUN_withWeights.ipynb**
    - GRU model with class weights for imbalanced data.
16. **GrunCnn.ipynb**
    - Hybrid GRU and CNN model for advanced sequence modeling.
17. **mini_plotter.py**
    - Python script for plotting EKG data from CSV or serial input.
18. **ML_Test.ipynb**
    - Miscellaneous machine learning experiments and tests.
19. **naiveBayes.ipynb**
    - Naive Bayes classifier for heartbeat classification and comparison with deep learning models.
20. **RandomForest.ipynb**
    - Random Forest classifier for heartbeat classification.
21. **RNN.ipynb**
    - Recurrent Neural Network (RNN/LSTM) for sequence-based heartbeat classification.
22. **SARIMA.ipynb**
    - Time series forecasting using SARIMA models.
23. **SARIMAWithRandomForest.ipynb**
    - Combines SARIMA time series modeling with Random Forest classification.
24. **Window.py**
    - Tkinter GUI for real-time EKG data recording, plotting, and CSV export.
25. **outputforfifteencbnnattention.png**
    - Example output image from CNN with attention model.
26. **class_weights.pkl**
    - Pickle file containing class weights for model training.
27. **multiclass_gru_model.h5**
    - Saved GRU model weights.
28. **EKGenie Midterm Presentation.pptx**
    - Project presentation slides.

## How to Use

### Jupyter Notebooks
- Open any notebook (e.g. `DataVisualization.ipynb`, `FeatureEngineering.ipynb`, `CNN.ipynb`, etc.) in Jupyter Notebook or VS Code.
- Run cells sequentially to load, preprocess, analyze, and model the data.

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
