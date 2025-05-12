# Capstone

## Project Overview
This project consists of multiple components for analyzing and visualizing EKG data. Below is a summary of the files included in this repository:

### Files

1. **HawraaJupyter.ipynb**
   - A Jupyter Notebook for data analysis and visualization of heartbeat datasets.
   - Includes tasks such as loading data, performing PCA, calculating correlation matrices, and visualizing results.

2. **Arduino.cpp**
   - Arduino code for reading EKG sensor data and transmitting it via serial communication.
   - Configured for a sampling rate of approximately 100Hz.

3. **Window.py**
   - A Python GUI application built with Tkinter for recording and visualizing EKG data in real-time.
   - Features include serial port connection, data recording, and live plotting.

## How to Use

### Jupyter Notebook
- Open `HawraaJupyter.ipynb` in Jupyter Notebook or VS Code.
- Follow the cells sequentially to load data, preprocess, and analyze it.

### Arduino Code
- Upload `Arduino.cpp` to an Arduino board connected to an EKG sensor.
- Ensure the correct analog pin and baud rate are configured.

### GUI Application
- Run `Window.py` using Python 3.
- Connect to the Arduino via the serial port and start recording EKG data.

## Requirements
- Python 3.x
- Libraries: `pandas`, `matplotlib`, `seaborn`, `numpy`, `serial`, `tkinter`
- Arduino IDE for uploading the Arduino code.

## License
This project is licensed under the MIT License.