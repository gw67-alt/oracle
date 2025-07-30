import matplotlib
matplotlib.use('TkAgg')  # or other backend as needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import deque
import threading
import time
import serial
from datetime import datetime, timedelta

class SerialSeismograph:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.02):
        """Initialize connection to real hardware"""
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        print(f"Connected to {port} at {baudrate} baud")
        
    def read_raw_data(self):
        """Read from serial and convert to magnitude"""
        try:
            line = self.serial.readline()
            if line:
                # Adjust parsing based on your device's output (e.g., comma-separated, JSON)
                magnitude = float(line.decode().strip())
                return abs(magnitude)
        except (ValueError, serial.SerialException, UnicodeDecodeError) as e:
            print(f"Serial error: {e}")
            return None

class MagnitudeMonitor:
    def __init__(self, port, baudrate=115200, sample_rate=50):
        # Connect to the actual seismometer
        self.seismograph = SerialSeismograph(port=port, baudrate=baudrate)
        self.sample_rate = sample_rate
        self.buffer_size = 200
        
        # Data buffer
        self.magnitude_data = deque(maxlen=self.buffer_size)
        self.running = False
        self.thread = None
        
        # Gantt chart tracking - ONLY magnitude calculation
        self.process_phases = {
            'Magnitude Calculation': {'active': False, 'start_time': None, 'duration': 0}
        }
        self.gantt_history = []
        
    def start_monitoring(self):
        """Start monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_activity, daemon=True)
        self.thread.start()
        
    def _monitor_activity(self):
        """Background thread for monitoring"""
        while self.running:
            magnitude = self.seismograph.read_raw_data()
            if magnitude is not None:
                self.magnitude_data.append(magnitude)
                self._process_magnitude_only(magnitude)
            
            # Wait a bit to avoid spinlock, but let hardware set the real rate
            # Remove sleep entirely if your device streams at fixed intervals
            time.sleep(1.0 / (self.sample_rate * 4))  
    
    def _process_magnitude_only(self, magnitude):
        """Process only magnitude calculation"""
        self._start_phase('Magnitude Calculation')
        time.sleep(0.002)  # Simulate processing time
        self._end_phase('Magnitude Calculation')
    
    def _start_phase(self, phase_name):
        """Start a processing phase"""
        if not self.process_phases[phase_name]['active']:
            self.process_phases[phase_name]['active'] = True
            self.process_phases[phase_name]['start_time'] = datetime.now()
    
    def _end_phase(self, phase_name):
        """End a processing phase"""
        if self.process_phases[phase_name]['active']:
            start_time = self.process_phases[phase_name]['start_time']
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Add to history
            self.gantt_history.append({
                'phase': phase_name,
                'start': start_time,
                'duration': duration,
                'end': end_time
            })
            
            # Keep only recent history (last 100 entries)
            if len(self.gantt_history) > 100:
                self.gantt_history.pop(0)
            
            self.process_phases[phase_name]['active'] = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

def draw_magnitude_gantt(ax, gantt_history):
    """Draw a Gantt chart for ONLY magnitude calculation"""
    ax.clear()
    
    if not gantt_history:
        ax.text(0.5, 0.5, "📊 MAGNITUDE CALCULATION GANTT\n\nWaiting for calculations...", 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7),
                transform=ax.transAxes)
        ax.set_title("📈 Magnitude Processing Timeline", fontsize=18, fontweight='bold')
        return
    
    # Only one phase color now
    phase_colors = {
        'Magnitude Calculation': '#d62728'
    }
    
    # Get recent history (last 20 seconds)
    current_time = datetime.now()
    recent_history = [h for h in gantt_history 
                     if (current_time - h['start']).total_seconds() <= 20]
    
    if not recent_history:
        ax.text(0.5, 0.5, "📊 No recent calculations", 
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        return
    
    # Draw Gantt bars for magnitude calculation only
    for entry in recent_history:
        phase = entry['phase']
        start_time = mdates.date2num(entry['start'])
        duration_days = entry['duration'] / 86400  # Convert seconds to days for matplotlib
        
        y_pos = 0  # Only one row now
        color = phase_colors[phase]
        
        # Draw the bar
        ax.barh(y_pos, duration_days, left=start_time, height=0.6, 
                alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
    
    # Format the chart
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(['Magnitude Calculation'])
    ax.set_xlabel('Time (Last 20 Seconds)', fontsize=12)
    ax.set_title('📈 Magnitude Calculation Timeline', fontsize=18, fontweight='bold')
    
    # Format x-axis to show time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=2))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add some statistics
    if recent_history:
        total_calcs = len(recent_history)
        avg_duration = np.mean([h['duration'] for h in recent_history])
        ax.text(0.02, 0.98, f"Total Calculations: {total_calcs}\nAvg Duration: {avg_duration:.4f}s", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

def main():
    print("📊 REAL-TIME MAGNITUDE CALCULATION GANTT CHART")
    print("=" * 50)
    
    # EDIT HERE: Set your actual serial port (e.g., 'COM3', '/dev/ttyACM0')
    port = '/dev/ttyUSB0'  # Change to your device's port
    baudrate = 115200      # Adjust to match your device
    
    monitor = MagnitudeMonitor(port=port, baudrate=baudrate, sample_rate=50)
    
    try:
        print("Starting seismic monitoring...")
        monitor.start_monitoring()
        
        time.sleep(2)  # Collect initial data
        
        # Create display with single plot
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        plt.show()
        
        print("📊 Magnitude-Only Gantt display is now active...")
        print("📈 Tracking only magnitude calculations from real hardware...")
        
        # Main loop
        for i in range(1000):  # Run for a while
            # Draw only the Gantt chart
            draw_magnitude_gantt(ax, monitor.gantt_history)
            
            plt.tight_layout()
            plt.pause(0.3)
            
            # Check if window was closed
            if not plt.get_fignums():
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.stop_monitoring()
        plt.ioff()
        plt.close('all')
        print("📴 System stopped.")

if __name__ == "__main__":
    main()
