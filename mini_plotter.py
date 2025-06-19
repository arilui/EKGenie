import tkinter as tk
from tkinter import ttk
import serial
import threading
import time
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Serial port settings
SERIAL_PORT = 'COM3'  # Change to your Arduino port
BAUD_RATE = 9600

class SerialPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("Arduino Serial Plotter")
        self.data = []
        self.max_points = 500

        self.last_peak_time = None
        self.heart_rates = []
        self.peak_threshold = 500  # Adjust this threshold for your signal
        self.min_peak_distance = 0.4  # Minimum time (in seconds) between peaks

        self.peak_detected = False

        # Set up matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_ylim(0, 1023)
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_title("Analog Input")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Voltage (mV)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Start serial thread
        self.running = True
        self.serial_thread = threading.Thread(target=self.read_serial)
        self.serial_thread.daemon = True
        self.serial_thread.start()

        self.update_plot()

    def read_serial(self):
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                while self.running:
                    line = ser.readline().decode('utf-8').strip()
                    if line.isdigit():
                        value = int(line)
                        self.data.append(value)
                        if len(self.data) > self.max_points:
                            self.data.pop(0)
        except serial.SerialException:
            print("Could not open serial port.")

    def update_plot(self):
        if self.data:
            self.line.set_data(range(len(self.data)), self.data)
            self.ax.set_xlim(0, self.max_points)
            self.ax.set_ylim(0, 1023)
        self.canvas.draw()
        self.root.after(50, self.update_plot)

    def on_close(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SerialPlotter(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

    