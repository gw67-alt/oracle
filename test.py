import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def moving_average(data, window_size=7):
    """Smooth data with a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

class OscilloscopeDisplay:
    def __init__(self, width=800, height=400, history_length=50):
        self.width = width
        self.height = height
        self.history = deque(maxlen=history_length)
        self.time_axis = np.linspace(0, 2*np.pi, width)
        
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
        
    def render(self):
        """Render the oscilloscope display"""
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
        if len(self.history) > 0:
            current_trace = self.history[-1]
            
            # Add glow effect by plotting multiple times with decreasing alpha and increasing width
            for i in range(3):
                self.ax.plot(range(self.width), current_trace, 
                           color=self.phosphor_green, alpha=0.2-i*0.05, linewidth=6+i*2)
            
            # Main bright trace on top
            self.ax.plot(range(self.width), current_trace, 
                       color=self.phosphor_green, alpha=1.0, linewidth=2)
        
        # Add oscilloscope-style title
        self.ax.text(0.02, 0.95, 'CURVATURE OSCILLOSCOPE', 
                    transform=self.ax.transAxes, color=self.phosphor_green,
                    fontsize=12, fontweight='bold', family='monospace')
        
        # Add technical readouts
        if len(self.history) > 0:
            current_data = self.history[-1]
            peak_val = np.max(np.abs(current_data))
            rms_val = np.sqrt(np.mean(current_data**2))
            
            self.ax.text(0.02, 0.05, f'PEAK: {peak_val:.3f}', 
                        transform=self.ax.transAxes, color=self.phosphor_green,
                        fontsize=10, family='monospace')
            self.ax.text(0.02, 0.12, f'RMS:  {rms_val:.3f}', 
                        transform=self.ax.transAxes, color=self.phosphor_green,
                        fontsize=10, family='monospace')
        
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
cap = cv2.VideoCapture(0)
oscilloscope = OscilloscopeDisplay(width=800, height=400, history_length=20)
smoother = CurvatureSmoother(maxlen=5)

print("Starting Oscilloscope-Style Curvature Analyzer...")
print("Press ESC to quit")

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
        
        if len(largest_contour) >= 20:
            # Calculate curvature
            curv = curvature(largest_contour, k=15)
            curv_smooth = moving_average(curv, window_size=9)
            
            # Add to temporal smoother
            smoother.add(curv_smooth)
            curv_time_smooth = smoother.get_smoothed()
            
            # Update oscilloscope with smoothed data
            oscilloscope.add_waveform(curv_time_smooth)
            
            # Draw contour on frame
            cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # Add info text
            cv2.putText(display_frame, f'Contour Points: {len(largest_contour)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Max Curvature: {np.max(curv_smooth):.4f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Render oscilloscope display
    scope_display = oscilloscope.render()
    
    # Show displays
    cv2.imshow('Green Object Detection', display_frame)
    cv2.imshow('Curvature Oscilloscope', scope_display)
    
    # Check for exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close('all')
print("Oscilloscope session ended.")
