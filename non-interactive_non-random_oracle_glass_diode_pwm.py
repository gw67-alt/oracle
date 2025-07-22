#look for voltage spikes
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import serial
import serial.tools.list_ports
import threading
import time
import json
import struct

def moving_average(data, window_size=7):
    """Smooth data with a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

class SerialReader:
    def __init__(self, port=None, baudrate=115200, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.running = False
        self.data_queue = deque(maxlen=1000)
        self.thread = None
        
    def list_ports(self):
        """List available serial ports"""
        ports = serial.tools.list_ports.comports()
        return [(port.device, port.description) for port in ports]
    
    def connect(self, port=None):
        """Connect to serial port"""
        if port:
            self.port = port
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False
    
    def start_reading(self):
        """Start reading data in a separate thread"""
        if self.serial_conn and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._read_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_reading(self):
        """Stop reading data"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _read_loop(self):
        """Main reading loop (runs in separate thread)"""
        buffer = ""
        
        while self.running and self.serial_conn:
            try:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._parse_data(line)
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Read error: {e}")
                break
    
    def _parse_data(self, line):
        """Parse incoming data line"""
        try:
            if line.replace('.', '').replace('-', '').isdigit():
                value = float(line)
                self.data_queue.append({'timestamp': time.time(), 'value': value, 'channel': 0})
            elif line.startswith('{'):
                data = json.loads(line)
                self.data_queue.append({'timestamp': time.time(), 'value': data.get('value', 0), 'channel': data.get('channel', 0)})
            elif ',' in line:
                parts = line.split(',')
                for i, part in enumerate(parts):
                    if part.strip().replace('.', '').replace('-', '').isdigit():
                        self.data_queue.append({'timestamp': time.time(), 'value': float(part.strip()), 'channel': i})
            elif ':' in line:
                key, value = line.split(':', 1)
                if value.strip().replace('.', '').replace('-', '').isdigit():
                    self.data_queue.append({'timestamp': time.time(), 'value': float(value.strip()), 'channel': hash(key.strip()) % 4, 'label': key.strip()})
        except Exception as e:
            print(f"Parse error for line '{line}': {e}")
    
    def get_latest_data(self, max_samples=100):
        """Get the latest data samples"""
        return list(self.data_queue)[-max_samples:]
    
    def get_channel_data(self, channel=0, max_samples=100):
        """Get data for a specific channel"""
        all_data = list(self.data_queue)
        channel_data = [d for d in all_data if d.get('channel', 0) == channel]
        return channel_data[-max_samples:]
    
    def disconnect(self):
        """Disconnect from serial port"""
        self.stop_reading()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.serial_conn = None

class OscilloscopeDisplay:
    def __init__(self, width=800, height=400, history_length=50, channels=4):
        self.width = width
        self.height = height
        self.channels = channels
        self.history = {i: deque(maxlen=history_length) for i in range(channels)}
        self.channel_colors = ['#00FF00', '#FF0080', '#00FFFF', '#FFFF00'] # Green, Pink, Cyan, Yellow
        
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        self.stats = {i: {'min': 0, 'max': 0, 'avg': 0, 'rms': 0} for i in range(channels)}

    def add_waveform(self, data, channel=0):
        """Add waveform data for a specific channel"""
        if len(data) == 0 or channel >= self.channels:
            return
            
        data = np.array(data)
        max_abs = np.max(np.abs(data))
        if max_abs > 0:
            normalized_data = data / max_abs * 1.5
        else:
            normalized_data = data
            
        if len(normalized_data) != self.width:
            if len(normalized_data) > 1:
                x_old = np.linspace(0, self.width-1, len(normalized_data))
                x_new = np.linspace(0, self.width-1, self.width)
                normalized_data = np.interp(x_new, x_old, normalized_data)
            else:
                normalized_data = np.full(self.width, normalized_data[0] if len(normalized_data) > 0 else 0)
        
        self.history[channel].append(normalized_data)
        
        if len(data) > 0:
            self.stats[channel] = {'min': np.min(data), 'max': np.max(data), 'avg': np.mean(data), 'rms': np.sqrt(np.mean(np.square(data)))}

    def render(self):
        """Render the oscilloscope display"""
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.grid(True, color='#333333', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axhline(y=0, color='#333333', linewidth=1, alpha=0.5)
        
        active_channels = 0
        for channel in range(self.channels):
            if not self.history[channel]:
                continue
            active_channels += 1
            color = self.channel_colors[channel % len(self.channel_colors)]
            
            for i, trace in enumerate(list(self.history[channel])[:-1]):
                alpha = 0.1 + (i / len(self.history[channel])) * 0.2
                self.ax.plot(range(self.width), trace + channel * 0.5, color=color, alpha=alpha, linewidth=1)
            
            current_trace = self.history[channel][-1]
            offset = channel * 0.5
            
            for i in range(3):
                self.ax.plot(range(self.width), current_trace + offset, color=color, alpha=0.2-i*0.05, linewidth=8+i*2)
            
            self.ax.plot(range(self.width), current_trace + offset, color=color, alpha=1.0, linewidth=2, label=f'CH{channel}')
        
        self.ax.text(0.02, 0.95, 'SERIAL ANALOG OSCILLOSCOPE', transform=self.ax.transAxes, color='#00FF00', fontsize=14, fontweight='bold', family='monospace')
        
        y_pos = 0.85
        for channel in range(self.channels):
            if not self.history[channel]:
                continue
            color = self.channel_colors[channel % len(self.channel_colors)]
            stats = self.stats[channel]
            self.ax.text(0.02, y_pos, f'CH{channel}: MIN:{stats["min"]:.2f}', transform=self.ax.transAxes, color=color, fontsize=9, family='monospace')
            y_pos -= 0.08
        
        self.ax.text(0.70, 0.95, f'Active Channels: {active_channels}', transform=self.ax.transAxes, color='#AAAAAA', fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.canvas.draw()
        
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        scope_img = buf.reshape(h, w, 4)[:, :, :3].copy()
        
        scope_img = self.add_scanlines(scope_img)
        return scope_img

    def add_scanlines(self, img):
        """Add CRT-style scanlines"""
        h, w = img.shape[:2]
        scanline_overlay = np.zeros_like(img, dtype=np.uint8)
        scanline_overlay[::3, :, :] = 20 # Dark lines
        img = cv2.subtract(img, scanline_overlay)
        return img
    
def main():
    serial_reader = SerialReader()
    oscilloscope = OscilloscopeDisplay(width=800, height=400, history_length=30, channels=4)
    
    print("=== Serial Analog Oscilloscope with Image Analysis ===")
    
    ports = serial_reader.list_ports()
    if not ports:
        print("No serial ports found!")
        return
    
    print("Available serial ports:")
    for i, (port, desc) in enumerate(ports):
        print(f"{i}: {port} - {desc}")
    
    try:
        port_idx = int(input(f"Select port (0-{len(ports)-1}): "))
        selected_port = ports[port_idx][0]
    except (ValueError, IndexError):
        selected_port = ports[0][0]
        print(f"Using default port: {selected_port}")
    
    try:
        baudrate = int(input("Enter baud rate (default 115200): ") or "115200")
    except ValueError:
        baudrate = 115200
    
    serial_reader.baudrate = baudrate
    
    if not serial_reader.connect(selected_port):
        return
    
    serial_reader.start_reading()
    print("Press ESC to quit")
    
    try:
        while True:
            # Step 1: Populate oscilloscope with serial data
            for channel in range(4):
                channel_data = serial_reader.get_channel_data(channel, max_samples=100)
                if channel_data:
                    values = [d['value'] for d in channel_data]
                    if len(values) > 0:
                        smoothed_values = moving_average(values, window_size=min(5, len(values)))
                        oscilloscope.add_waveform(smoothed_values, channel=channel)

            # Step 2: Render the oscilloscope data into an image
            scope_display_img = oscilloscope.render()

            # --- Step 3: Apply Image Processing Logic to the Rendered Image ---
            # Convert the rendered image to HSV color space for analysis
            hsv_img = cv2.cvtColor(scope_display_img, cv2.COLOR_BGR2HSV)
            
            # Define HSV color range for the green oscilloscope trace (Channel 0)
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            
            # Create a mask to isolate the green parts of the image
            mask = cv2.inRange(hsv_img, lower_green, upper_green)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            
            # Find contours (the shapes of the waveform) in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            knot_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 50: # Filter out small noise
                    # Draw detected contours onto the display image in red
                    cv2.drawContours(scope_display_img, [cnt], -1, (0, 0, 255), 2)
                    knot_count += 1

            # Display the number of detected contours (wave segments) on the image
            cv2.putText(scope_display_img, f'Wave Segments: {knot_count}', (550, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- Step 4: Display the Final Image ---
            cv2.imshow('Serial Oscilloscope with Analysis', scope_display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                for ch in range(4):
                    oscilloscope.history[ch].clear()
                print("Display cleared")

            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        serial_reader.disconnect()
        cv2.destroyAllWindows()
        plt.close('all')
        print("Serial oscilloscope session ended.")

if __name__ == "__main__":
    main()
