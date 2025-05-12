import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import serial.tools.list_ports
import threading
import time
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class EKGRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EKGenie - EKG Recorder")
        self.root.geometry("900x600")
        
        # Serial connection variables
        self.serial_port = None
        self.serial_connected = False
        self.recording = False
        self.data = []
        self.time_data = []
        self.start_time = 0
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize plot
        self.setup_plot()
        
        # Check for available ports
        self.update_ports()
        
    def create_widgets(self):
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=(10, 5))
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Port selection
        ttk.Label(control_frame, text="Port:").grid(row=0, column=0, sticky=tk.W)
        self.port_combobox = ttk.Combobox(control_frame, state="readonly")
        self.port_combobox.grid(row=0, column=1, padx=5)
        
        # Baud rate selection
        ttk.Label(control_frame, text="Baud Rate:").grid(row=0, column=2, sticky=tk.W)
        self.baud_combobox = ttk.Combobox(control_frame, values=[9600, 19200, 38400, 57600, 115200], state="readonly")
        self.baud_combobox.grid(row=0, column=3, padx=5)
        self.baud_combobox.set(115200)  # Default baud rate
        
        # Connect button
        self.connect_button = ttk.Button(control_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=4, padx=5)
        
        # Refresh ports button
        ttk.Button(control_frame, text="Refresh Ports", command=self.update_ports).grid(row=0, column=5, padx=5)
        
        # Recording controls
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.record_button.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        
        self.save_button = ttk.Button(control_frame, text="Save Data", command=self.save_data, state=tk.DISABLED)
        self.save_button.grid(row=1, column=2, columnspan=2, pady=(10, 0))
        
        self.clear_button = ttk.Button(control_frame, text="Clear Plot", command=self.clear_plot, state=tk.DISABLED)
        self.clear_button.grid(row=1, column=4, columnspan=2, pady=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Disconnected")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=6, pady=(10, 0), sticky=tk.W)
        
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
        self.ax.set_xlim(0, 10)  # Initial x-axis limit (10 seconds)
        self.ax.set_ylim(-1, 1)  # Adjust based on your EKG signal range
        
    def update_ports(self):
        """Update the list of available serial ports"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combobox['values'] = ports
        if ports:
            self.port_combobox.set(ports[0])
            
    def toggle_connection(self):
        """Connect or disconnect from the serial port"""
        if self.serial_connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        """Connect to the selected serial port"""
        port = self.port_combobox.get()
        baud = self.baud_combobox.get()
        
        if not port:
            messagebox.showerror("Error", "No port selected!")
            return
            
        try:
            self.serial_port = serial.Serial(port, int(baud), timeout=1)
            self.serial_connected = True
            self.connect_button.config(text="Disconnect")
            self.record_button.config(state=tk.NORMAL)
            self.status_var.set(f"Connected to {port} at {baud} baud")
            
            # Start a thread to read serial data
            self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.read_thread.start()
            
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", str(e))
            
    def disconnect(self):
        """Disconnect from the serial port"""
        if self.recording:
            self.toggle_recording()  # Stop recording if active
            
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            
        self.serial_connected = False
        self.connect_button.config(text="Connect")
        self.record_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.status_var.set("Disconnected")
        
    def toggle_recording(self):
        """Start or stop recording data"""
        if self.recording:
            self.recording = False
            self.record_button.config(text="Start Recording")
            self.save_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.status_var.set("Recording stopped - Ready to save data")
        else:
            self.recording = True
            self.data = []
            self.time_data = []
            self.start_time = time.time()
            self.record_button.config(text="Stop Recording")
            self.save_button.config(state=tk.DISABLED)
            self.clear_button.config(state=tk.DISABLED)
            self.status_var.set("Recording...")
            
    def read_serial(self):
        """Read data from the serial port in a separate thread"""
        while self.serial_connected and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    try:
                        value = float(line)
                        current_time = time.time() - self.start_time
                        
                        # Update the data arrays if recording
                        if self.recording:
                            self.time_data.append(current_time)
                            self.data.append(value)
                            
                            # Update plot every N points for performance
                            if len(self.time_data) % 10 == 0:
                                self.update_plot()
                                
                    except ValueError:
                        pass  # Ignore lines that can't be converted to float
                        
            except serial.SerialException:
                break
            except UnicodeDecodeError:
                continue
                
        # If we get here, the serial connection was lost
        if self.serial_connected:  # Only if we didn't intentionally disconnect
            self.root.after(0, self.disconnect)
            
    def update_plot(self):
        """Update the plot with new data"""
        if not self.time_data or not self.data:
            return
            
        self.line.set_data(self.time_data, self.data)
        
        # Adjust x-axis limits to show the most recent 10 seconds
        current_time = self.time_data[-1]
        if current_time > 10:
            self.ax.set_xlim(current_time - 10, current_time)
        else:
            self.ax.set_lim(0, 10)
            
        # Auto-scale y-axis with some padding
        y_min, y_max = min(self.data), max(self.data)
        padding = max(0.1 * (y_max - y_min), 0.5)  # At least 0.5 units padding
        self.ax.set_ylim(y_min - padding, y_max + padding)
        
        # Redraw the canvas
        self.canvas.draw()
        
    def save_data(self):
        """Save the recorded data to a CSV file"""
        if not self.time_data or not self.data:
            messagebox.showwarning("No Data", "No data to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save EKG Data"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Time (s)", "Voltage (mV)"])
                    for t, v in zip(self.time_data, self.data):
                        writer.writerow([t, v])
                        
                messagebox.showinfo("Success", f"Data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                
    def clear_plot(self):
        """Clear the plot and recorded data"""
        self.data = []
        self.time_data = []
        self.line.set_data([], [])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-1, 1)
        self.canvas.draw()
        self.status_var.set("Plot cleared")
        
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