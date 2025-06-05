import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time
import hashlib
import queue

def moving_average(data, window_size=7):
    """Smooth data with a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def sha256_proof_of_work(data,i, difficulty=4):

    """
    Simple proof of work function using SHA-256.
    Returns True if hash starts with 'difficulty' number of zeros.
    """
    if len(data) == 0:
        return False
    for j in range(100000):
        # Convert data to string for hashing
        data_str = data + str(i+j)
        hash_result = hashlib.sha256(data_str.encode()).hexdigest()
    
        # Check if hash starts with required number of zeros
   
        if hash_result.startswith('0' * difficulty):
            print(data_str, hash_result)
            return hash_result.startswith('0' * difficulty)
    return False
class ProofOfWorkThread(threading.Thread):
    def __init__(self, data_queue, result_queue):
        super().__init__()
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.counter = 0
        self.running = True
        self.daemon = True
        self.credit = 10000000000
    def run(self):
        """Main thread loop - counts by 1000 and checks proof of work"""
        while self.running:
            try:
                # Get latest oscilloscope data (non-blocking)
                current_data = None
                try:
                    current_data = self.data_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Increment counter by 1000
                self.counter += 100000
                
                # Check if we have valid data (not NaN, not empty)
                if current_data is not None and len(current_data) > 0:
                    # Check for NaN values
                    if np.any(np.isnan(current_data)) and self.credit > 0:
                        self.credit -= 1
                    if not np.any(np.isnan(current_data)) and self.credit > 0:
                        self.credit += 1
                        # Data is optimal - check proof of work
                        pow_result = sha256_proof_of_work("GeorgeW",self.counter, difficulty=5)
                        if pow_result == True:
                            # Send result back to main thread
                            result_data = {
                                'counter': self.counter,
                                'data_valid': True,
                                'proof_of_work': pow_result,
                                'data_hash': hashlib.sha256(','.join([f"{x:.6f}" for x in current_data]).encode()).hexdigest()[:16]
                            }
                            
                            try:
                                self.result_queue.put_nowait(result_data)
                            except queue.Full:
                                pass  # Skip if queue is full
                    else:
                        # Data contains NaN
                        result_data = {
                            'counter': self.counter,
                            'data_valid': False,
                            'proof_of_work': False,
                            'data_hash': 'NaN_DATA'
                        }
                        
                        try:
                            self.result_queue.put_nowait(result_data)
                        except queue.Full:
                            pass
                
                # Sleep to control thread frequency
                time.sleep(0.001)  # 10 Hz
                
            except Exception as e:
                print(f"PoW Thread Error: {e}")
                time.sleep(0.1)
    
    def stop(self):
        self.running = False

class OscilloscopeDisplay:
    def __init__(self, width=800, height=400, history_length=50):
        self.width = width
        self.height = height
        self.history = deque(maxlen=history_length)
        self.time_axis = np.linspace(0, 2*np.pi, width)
        
        # Threading components
        self.data_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=20)
        self.pow_thread = ProofOfWorkThread(self.data_queue, self.result_queue)
        self.pow_thread.start()
        
        # Proof of work results
        self.latest_pow_result = None
        
        # Set up matplotlib with oscilloscope style
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Oscilloscope grid
        self.ax.grid(True, color='#00FF00', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(-2, 2)
        
        # Remove ticks and labels for clean look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Phosphor green color
        self.phosphor_green = '#00FF00'
        self.dim_green = '#004400'
        self.warning_orange = '#FF8800'
        self.success_cyan = '#00FFFF'
        
    def add_waveform(self, data):
        """Add new curvature data to the oscilloscope display"""
        if len(data) == 0:
            return
            
        # Normalize data to fit display range
        if np.max(np.abs(data)) > 0:
            normalized_data = data / np.max(np.abs(data)) * 1.5
        else:
            normalized_data = data
            
        # Resample to fit display width
        if len(normalized_data) != self.width:
            x_old = np.linspace(0, self.width-1, len(normalized_data))
            x_new = np.linspace(0, self.width-1, self.width)
            normalized_data = np.interp(x_new, x_old, normalized_data)
            
        self.history.append(normalized_data)
        
        # Send data to proof of work thread
        try:
            self.data_queue.put_nowait(normalized_data.copy())
        except queue.Full:
            pass  # Skip if queue is full
        
    def check_pow_results(self):
        """Check for new proof of work results"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.latest_pow_result = result
        except queue.Empty:
            pass  # No new results
        
    def render(self):
        """Render the oscilloscope display"""
        # Check for new proof of work results
        self.check_pow_results()
        
        self.ax.clear()
        
        # Set up oscilloscope appearance
        self.ax.set_facecolor('black')
        self.ax.grid(True, color=self.dim_green, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add center line
        self.ax.axhline(y=0, color=self.dim_green, linewidth=1, alpha=0.5)
        
        # Draw historical traces (fading)
        if len(self.history) > 1:
            for i, trace in enumerate(list(self.history)[:-1]):
                alpha = 0.1 + (i / len(self.history)) * 0.3
                self.ax.plot(range(self.width), trace, 
                           color=self.phosphor_green, alpha=alpha, linewidth=1)
        
        # Draw current trace (brightest) with glow effect
        trace_color = self.phosphor_green
        if self.latest_pow_result and self.latest_pow_result['proof_of_work']:
            trace_color = self.success_cyan  # Cyan for successful proof of work
        elif self.latest_pow_result and not self.latest_pow_result['data_valid']:
            trace_color = self.warning_orange  # Orange for invalid data
            
        if len(self.history) > 0:
            current_trace = self.history[-1]
            
            # Add glow effect by plotting multiple times with decreasing alpha and increasing width
            for i in range(3):
                self.ax.plot(range(self.width), current_trace, 
                           color=trace_color, alpha=0.2-i*0.05, linewidth=6+i*2)
            
            # Main bright trace on top
            self.ax.plot(range(self.width), current_trace, 
                       color=trace_color, alpha=1.0, linewidth=2)
        
        # Add oscilloscope-style title
        title_color = self.phosphor_green
        if self.latest_pow_result and self.latest_pow_result['proof_of_work']:
            title_color = self.success_cyan
            
        self.ax.text(0.02, 0.95, 'CURVATURE OSCILLOSCOPE [POW ENABLED]', 
                    transform=self.ax.transAxes, color=title_color,
                    fontsize=12, fontweight='bold', family='monospace')
        
        # Add technical readouts
        y_pos = 0.85
        if len(self.history) > 0:
            current_data = self.history[-1]
            peak_val = np.max(np.abs(current_data))
            rms_val = np.sqrt(np.mean(current_data**2))
            
            self.ax.text(0.02, y_pos, f'PEAK: {peak_val:.3f}', 
                        transform=self.ax.transAxes, color=self.phosphor_green,
                        fontsize=10, family='monospace')
            y_pos -= 0.07
            self.ax.text(0.02, y_pos, f'RMS:  {rms_val:.3f}', 
                        transform=self.ax.transAxes, color=self.phosphor_green,
                        fontsize=10, family='monospace')
            y_pos -= 0.07
        
        # Add proof of work status
        if self.latest_pow_result:
            pow_color = self.success_cyan if self.latest_pow_result['proof_of_work'] else self.phosphor_green
            valid_color = self.phosphor_green if self.latest_pow_result['data_valid'] else self.warning_orange
            
            self.ax.text(0.02, y_pos, f'COUNT: {self.latest_pow_result["counter"]:,}', 
                        transform=self.ax.transAxes, color=self.phosphor_green,
                        fontsize=10, family='monospace')
            y_pos -= 0.07
            
            self.ax.text(0.02, y_pos, f'VALID: {"YES" if self.latest_pow_result["data_valid"] else "NO"}', 
                        transform=self.ax.transAxes, color=valid_color,
                        fontsize=10, family='monospace')
            y_pos -= 0.07
            
            self.ax.text(0.02, y_pos, f'POW:   {"PASS" if self.latest_pow_result["proof_of_work"] else "FAIL"}', 
                        transform=self.ax.transAxes, color=pow_color,
                        fontsize=10, family='monospace')
            y_pos -= 0.07
            
            self.ax.text(0.02, y_pos, f'HASH:  {self.latest_pow_result["data_hash"]}', 
                        transform=self.ax.transAxes, color=self.phosphor_green,
                        fontsize=8, family='monospace')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert to image
        self.fig.canvas.draw()
        # Use buffer_rgba() for newer matplotlib versions
        try:
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height()
            scope_img = buf.reshape(h, w, 4)[:, :, :3]  # Remove alpha channel
        except AttributeError:
            # Fallback for older matplotlib versions
            try:
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                scope_img = buf.reshape(h, w, 3)
            except AttributeError:
                # Last resort - use tostring_argb
                buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                scope_img = buf.reshape(h, w, 4)[:, :, 1:4]  # Extract RGB from ARGB
        
        # Add scanline effect
        scope_img = self.add_scanlines(scope_img)
        
        return scope_img
    
    def add_scanlines(self, img):
        """Add CRT-style scanlines for authentic oscilloscope look"""
        h, w = img.shape[:2]
        for y in range(0, h, 3):
            if y < h:
                img[y, :] = img[y, :] * 0.8
        return img
    
    def cleanup(self):
        """Clean up threading resources"""
        if hasattr(self, 'pow_thread'):
            self.pow_thread.stop()
            self.pow_thread.join(timeout=1.0)

class CurvatureSmoother:
    def __init__(self, maxlen=5):
        self.history = deque(maxlen=maxlen)
    
    def add(self, curv):
        self.history.append(curv)
    
    def get_smoothed(self):
        if not self.history:
            return []
        max_len = max(len(c) for c in self.history)
        padded = [np.pad(c, (0, max_len - len(c)), 'constant') for c in self.history]
        avg = np.mean(padded, axis=0)
        return avg

def curvature(contour, k=15):
    n = len(contour)
    curvatures = []
    for i in range(n):
        p1 = contour[i % n][0]
        p2 = contour[(i - k) % n][0]
        p3 = contour[(i + k) % n][0]
        
        mat = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1]
        ])
        
        A = np.linalg.det(mat)
        if abs(A) < 1e-6:
            curvatures.append(0)
            continue
            
        Dx = np.linalg.det(np.array([
            [p1[0]**2 + p1[1]**2, p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[1], 1]
        ]))
        
        Dy = -np.linalg.det(np.array([
            [p1[0]**2 + p1[1]**2, p1[0], 1],
            [p2[0]**2 + p2[1]**2, p2[0], 1],
            [p3[0]**2 + p3[1]**2, p3[0], 1]
        ]))
        
        xc = Dx / (2*A)
        yc = Dy / (2*A)
        r = np.sqrt((p1[0] - xc)**2 + (p1[1] - yc)**2)
        curvature_val = 1/r if r != 0 else 0
        curvatures.append(curvature_val)
    
    return np.array(curvatures)

# Main execution
def main():
    cap = cv2.VideoCapture(0)
    oscilloscope = OscilloscopeDisplay(width=800, height=400, history_length=20)
    smoother = CurvatureSmoother(maxlen=5)

    print("Starting Threaded Oscilloscope-Style Curvature Analyzer with Proof of Work...")
    print("- Counter increments by 1000 each cycle")
    print("- SHA-256 proof of work validates optimal data")
    print("- Green trace = normal, Cyan = PoW success, Orange = invalid data")
    print("Press ESC to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame for green objects
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([36, 25, 25])
            upper_green = np.array([86, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Prepare display frame
            display_frame = frame.copy()
            
            # Process largest contour
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if len(largest_contour) >= 10:
                    # Calculate curvature
                    curv = curvature(largest_contour, k=15)
                    curv_smooth = moving_average(curv, window_size=9)
                    
                    # Add to temporal smoother
                    smoother.add(curv_smooth)
                    curv_time_smooth = smoother.get_smoothed()
                    
                    # Update oscilloscope with smoothed data
                    oscilloscope.add_waveform(abs(-1/curv_time_smooth))
                    
                    # Draw contour on frame
                    cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)
                    
                    # Add info text
                    cv2.putText(display_frame, f'Contour Points: {len(largest_contour)}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f'Max Curvature: {np.max(curv_smooth):.4f}', 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add PoW status to camera feed
                    if hasattr(oscilloscope, 'latest_pow_result') and oscilloscope.latest_pow_result:
                        pow_status = "PoW: PASS" if oscilloscope.latest_pow_result['proof_of_work'] else "PoW: FAIL"
                        color = (0, 255, 255) if oscilloscope.latest_pow_result['proof_of_work'] else (0, 255, 0)
                        cv2.putText(display_frame, pow_status, 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(display_frame, f'Count: {oscilloscope.latest_pow_result["counter"]:,}', 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Render oscilloscope display
            scope_display = oscilloscope.render()
            
            # Show displays
            cv2.imshow('Green Object Detection', display_frame)
            cv2.imshow('Curvature Oscilloscope', scope_display)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        oscilloscope.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        print("Threaded oscilloscope session ended.")

if __name__ == "__main__":
    main()
