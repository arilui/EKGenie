import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import threading
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow import keras
from CNN import FocalLoss
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.signal import butter, filtfilt, find_peaks

custom_objects = {'FocalLoss': FocalLoss}
cnn = keras.models.load_model('cnn_model.keras', custom_objects=custom_objects)
print("imports done")

conditions = {
    0: "Normal (N)",
    1: "Supraventricular premature (S)",
    2: "Premature ventricular contraction (V)",
    3: "Fusion of ventricular and normal beat (F)",
    4: "Unclassifiable beat (Q)",
    5: "Abnormal (A)"
}

class EKGRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EKGenie - EKG Recorder")
        self.root.geometry("900x600")
        
        # Data storage - using NumPy arrays
        self.time_data = np.array([], dtype=np.float32)
        self.voltage_data = np.array([], dtype=np.float32)
        
        # Serial connection variables
        self.serial_port = None
        self.serial_connected = False
        self.recording = False
        self.start_time = 0
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize plot
        self.setup_plot()
        
    def create_widgets(self):
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=(10, 5))
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Port label (fixed to COM3)
        ttk.Label(control_frame, text="Port: COM3").grid(row=0, column=0, sticky=tk.W)
        
        # Baud rate selection
        ttk.Label(control_frame, text="Baud Rate:").grid(row=0, column=1, sticky=tk.W)
        self.baud_combobox = ttk.Combobox(control_frame, values=[9600, 19200, 38400, 57600, 115200], state="readonly")
        self.baud_combobox.grid(row=0, column=2, padx=5)
        self.baud_combobox.set(9600)  # Default baud rate
        
        # Connect button
        self.connect_button = ttk.Button(control_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=3, padx=5)
        
        # Recording controls
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.record_button.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        
        # Changed from Save to Export option
        self.export_button = ttk.Button(control_frame, text="Export Data", command=self.export_data, state=tk.DISABLED)
        self.export_button.grid(row=1, column=2, columnspan=2, pady=(10, 0))
        
        self.clear_button = ttk.Button(control_frame, text="Clear Data", command=self.clear_data, state=tk.DISABLED)
        self.clear_button.grid(row=1, column=4, columnspan=2, pady=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Disconnected")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.W)
        
        # Heart rate display
        self.hr_var = tk.StringVar()
        self.hr_var.set("Heart Rate: -- BPM")
        ttk.Label(control_frame, textvariable=self.hr_var).grid(row=2, column=3, columnspan=3, pady=(10, 0), sticky=tk.E)
        
        # Plot frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("EKG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_plot(self):
        """Initialize the plot with empty data"""
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlim(0, 500)  # Show last 500 samples (adjust as needed)
        self.ax.set_ylim(300, 500)  # Fixed y-axis range
        self.ax.set_xlabel("Sample Index")  # Changed from Time (s)
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.set_title("EKG Signal")
        self.ax.grid(True)
            
    def toggle_connection(self):
        """Connect or disconnect from the serial port"""
        if self.serial_connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        """Connect to COM3"""
        baud = self.baud_combobox.get()
        
        try:
            self.serial_port = serial.Serial('COM3', int(baud), timeout=1)
            self.serial_connected = True
            self.connect_button.config(text="Disconnect")
            self.record_button.config(state=tk.NORMAL)
            self.status_var.set(f"Connected to COM3 at {baud} baud")
            
            # Start a thread to read serial data
            self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.read_thread.start()
            
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to COM3: {str(e)}")
            
    def disconnect(self):
        """Disconnect from the serial port"""
        if self.recording:
            self.toggle_recording()  # Stop recording if active
            
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            
        self.serial_connected = False
        self.connect_button.config(text="Connect")
        self.record_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.status_var.set("Disconnected")
        self.hr_var.set("Heart Rate: -- BPM")
    
    def analyze_ekg_data(self):
        """Analyze the recorded EKG data using the CNN"""
        if len(self.voltage_data) == 0:
            messagebox.showwarning("No Data", "No EKG data to analyze!")
            return
            
        try:
            print("Length of voltage data:", len(self.voltage_data))
            print(len(self.voltage_data)/(self.time_data[-1]-self.time_data[0]), "samples per second")


            # Group the voltage data into segments of 133 samples
            segment_length = 133
            num_segments = len(self.voltage_data) // segment_length
            if num_segments == 0:
                messagebox.showwarning("Not Enough Data", "Not enough data for analysis (need at least 133 samples).")
                return

            # Truncate data to a multiple of segment_length
            truncated_data = self.voltage_data[:num_segments * segment_length]

            # Reshape for CNN: (num_segments, 133, 1)
            data = truncated_data.reshape((num_segments, segment_length, 1))
            
            # Normalize each segment
            scaler = StandardScaler()
            data = scaler.fit_transform(data.reshape(-1, segment_length)).reshape(num_segments, segment_length, 1)


            diagnosis = cnn.predict(data)
            predictions = np.argmax(diagnosis, axis=1)
            predictions = [conditions[p] for p in predictions]
            # Count unique instances of each condition
            condition_counts = Counter(predictions)
            counts_str = "\n".join(f"{cond}: {count}" for cond, count in condition_counts.items())
            
            
            # Show results to user
            result_message = (
                f"Analysis Results:\n\n"
                f"Diagnosis: {counts_str}\n"
            )
            
            messagebox.showinfo("Analysis Complete", result_message)
            
            # Update status with diagnosis
            self.status_var.set(
                f"Analysis complete - {counts_str}"
            )
            
        except Exception as e:
            messagebox.showerror("Analysis Error", 
                               f"Failed to analyze EKG data:\n{str(e)}")
            self.status_var.set("Analysis failed")
        
    def toggle_recording(self):
        """Start or stop recording data"""
        if self.recording:
            # Stop recording
            self.recording = False
            self.record_button.config(text="Start Recording")
            self.export_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.status_var.set(f"Recording stopped - {len(self.time_data)} samples collected\nTime elapsed: {time.time() - self.start_time:.2f} seconds")
            
            # Analyze the recorded data
            self.analyze_ekg_data()
        else:
            # Start new recording
            self.recording = True
            self.time_data = np.array([], dtype=np.float32)
            self.voltage_data = np.array([], dtype=np.float32)
            self.start_time = time.time()
            self.record_button.config(text="Stop Recording")
            self.export_button.config(state=tk.DISABLED)
            self.clear_button.config(state=tk.DISABLED)
            self.status_var.set("Recording...")
            self.hr_var.set("Heart Rate: -- BPM")
            
    def find_r_peaks(self, threshold=400, min_distance=0.1):
        """Detect R-peaks in the EKG data"""
        peaks = []
        if len(self.voltage_data) < 3:
            return peaks
            
        # Simple peak detection (adjusted for 0-1000 mV range)
        for i in range(1, len(self.voltage_data)-1):
            if (self.voltage_data[i] > self.voltage_data[i-1] and 
                self.voltage_data[i] > self.voltage_data[i+1] and 
                self.voltage_data[i] > threshold):
                # Check minimum distance between peaks
                if not peaks or (self.time_data[i] - peaks[-1][0]) > min_distance:
                    peaks.append((self.time_data[i], self.voltage_data[i]))
        
        return peaks
    
    def calculate_heart_rate(self, peaks):
        """Calculate heart rate from R-peak times"""
        if len(peaks) < 2:
            return 0
        
        peaks = np.array(peaks)
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(peaks[:, 0])
        
        # Calculate average RR interval
        avg_rr = np.mean(rr_intervals)
        
        # Calculate heart rate (beats per minute)
        return 60 / avg_rr
            
    def read_serial(self):
        """Read data from the serial port in a separate thread"""
        while self.serial_connected and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    try:
                        value = float(line)
                        current_time = time.time() - self.start_time
                        
                        if self.recording:
                            # Append to numpy arrays
                            self.time_data = np.append(self.time_data, current_time)
                            self.voltage_data = np.append(self.voltage_data, value)
                            
                            # Update plot periodically
                            if len(self.time_data) % 10 == 0:
                                self.update_plot()
                                
                    except ValueError:
                        pass  # Ignore lines that can't be converted to float
                        
            except serial.SerialException:
                break
            except UnicodeDecodeError:
                continue
                
        # If we get here, the serial connection was lost
        if self.serial_connected:
            self.root.after(0, self.disconnect)
            
    def update_plot(self):
        """Update the plot with new data"""
        if len(self.time_data) == 0 or len(self.voltage_data) == 0:
            return
            
        sample_indices = np.arange(len(self.voltage_data))
        self.line.set_data(sample_indices, self.voltage_data)
        
        # Detect R-peaks (using 400 mV threshold)
        peaks = self.find_r_peaks(threshold=400)
        
        # Clear previous peak markers
        for artist in self.ax.lines[1:]:
            artist.remove()

        
        # Calculate and display heart rate
        heart_rate = self.calculate_heart_rate(peaks)
        if heart_rate > 0:
            hr_text = f"Heart Rate: {heart_rate:.1f} BPM"
            if not hasattr(self, 'hr_text'):
                self.hr_text = self.ax.text(0.02, 0.95, hr_text, 
                                          transform=self.ax.transAxes,
                                          bbox=dict(facecolor='white', alpha=0.8))
            else:
                self.hr_text.set_text(hr_text)
            self.hr_var.set(f"Heart Rate: {heart_rate:.1f} BPM")
        else:
            self.hr_var.set("Heart Rate: -- BPM")
        
        # Adjust x-axis limits to show the most recent 10 seconds
        
        if len(sample_indices) > 450:
            self.ax.set_xlim(len(sample_indices) - 450, len(sample_indices))
        else:
            self.ax.set_xlim(0, 450)
        
        # Maintain fixed y-axis range (0-1000 mV)
        self.ax.set_ylim(300, 450)
        
        # Redraw the canvas
        self.canvas.draw()
        
    def export_data(self):
        """Optionally export data to file"""
        if len(self.time_data) == 0:
            messagebox.showwarning("No Data", "No data to export!")
            return
            
        choice = messagebox.askyesnocancel("Export Data", 
                                          "Export recorded data to file?\n\n"
                                          "Yes: Save to NPZ\n"
                                          "No: Copy to clipboard\n"
                                          "Cancel: Do nothing")
        
        if choice is None:  # Cancel
            return
        elif choice:  # Yes - Save to NPZ
            file_path = filedialog.asksaveasfilename(
                defaultextension=".npz",
                filetypes=[("NumPy compressed files", "*.npz"), ("All files", "*.*")],
                title="Save EKG Data"
            )
            
            if file_path:
                try:
                    np.savez_compressed(
                        file_path,
                        time=self.time_data,
                        voltage=self.voltage_data
                    )
                    messagebox.showinfo("Success", f"Data exported to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export file: {str(e)}")
        else:  # No - Copy to clipboard
            try:
                # Create a string representation of the data
                data_str = "Time(s)\tVoltage(mV)\n"
                for t, v in zip(self.time_data, self.voltage_data):
                    data_str += f"{t:.3f}\t{v:.2f}\n"
                
                self.root.clipboard_clear()
                self.root.clipboard_append(data_str)
                messagebox.showinfo("Success", "Data copied to clipboard")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy data: {str(e)}")
                
    def clear_data(self):
        """Clear the stored data and plot"""
        self.time_data = np.array([], dtype=np.float32)
        self.voltage_data = np.array([], dtype=np.float32)
        self.line.set_data([], [])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(300, 450)
        
        # Clear peak markers and heart rate text
        for artist in self.ax.lines[1:]:
            artist.remove()
        if hasattr(self, 'hr_text'):
            self.hr_text.remove()
            del self.hr_text
            
        self.canvas.draw()
        self.status_var.set("Data cleared")
        self.hr_var.set("Heart Rate: -- BPM")
        self.export_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        
    def on_closing(self):
        """Handle window closing event"""
        if self.serial_connected:
            self.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EKGRecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()