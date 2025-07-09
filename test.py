import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

def moving_average(data, window_size=7):
    """Smooth data with a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def detect_sharp_signals_mid_graph(curvature_data, threshold=0.8, min_prominence=0.3, mid_range_percent=0.4):
    """
    Detect sharp signals specifically in the middle portion of the curvature data.
    
    Args:
        curvature_data: Array of curvature values
        threshold: Minimum value to consider as a sharp signal
        min_prominence: Minimum prominence for peak detection
        mid_range_percent: Percentage of data length to consider as "middle" (0.4 = 40%)
    
    Returns:
        Dictionary with sharp signal information focused on mid-graph region
    """
    if len(curvature_data) == 0:
        return {"indices": [], "values": [], "count": 0, "mid_range": (0, 0)}
    
    # Calculate middle range indices
    data_length = len(curvature_data)
    mid_start = int(data_length * (0.5 - mid_range_percent/2))
    mid_end = int(data_length * (0.5 + mid_range_percent/2))
    
    # Ensure bounds are valid
    mid_start = max(1, mid_start)
    mid_end = min(data_length - 1, mid_end)
    
    sharp_indices = []
    sharp_values = []
    
    # Only search in the middle range
    for i in range(mid_start, mid_end):
        current_val = curvature_data[i]
        
        # Check if current point is above threshold
        if current_val > threshold:
            # Check if it's a local maximum with sufficient prominence
            left_val = curvature_data[i-1]
            right_val = curvature_data[i+1]
            
            if (current_val > left_val and current_val > right_val and 
                min(current_val - left_val, current_val - right_val) > min_prominence):
                sharp_indices.append(i)
                sharp_values.append(current_val)
    
    return {
        "indices": sharp_indices,
        "values": sharp_values,
        "count": len(sharp_indices),
        "max_value": max(sharp_values) if sharp_values else 0,
        "avg_value": np.mean(sharp_values) if sharp_values else 0,
        "mid_range": (mid_start, mid_end),
        "mid_range_percent": mid_range_percent * 100
    }

class OscilloscopeDisplay:
    def __init__(self, width=800, height=400, history_length=50):
        self.width = width
        self.height = height
        self.history = deque(maxlen=history_length)
        self.time_axis = np.linspace(0, 2*np.pi, width)
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.grid(True, color='#00FF00', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.phosphor_green = '#00FF00'
        self.dim_green = '#004400'

    def add_waveform(self, data):
        if len(data) == 0:
            return
        if np.max(np.abs(data)) > 0:
            normalized_data = data / np.max(np.abs(data)) * 1.5
        else:
            normalized_data = data
        if len(normalized_data) != self.width:
            x_old = np.geomspace(1, self.width-1, len(normalized_data))
            x_new = np.geomspace(1, self.width-1, self.width)
            normalized_data = np.interp(x_new, x_old, normalized_data)
        self.history.append(normalized_data)

    def render(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.grid(True, color=self.dim_green, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axhline(y=0, color=self.dim_green, linewidth=1, alpha=0.5)
        
        if len(self.history) > 1:
            for i, trace in enumerate(list(self.history)[:-1]):
                alpha = 0.1 + (i / len(self.history)) * 0.3
                self.ax.plot(range(self.width), trace, color=self.phosphor_green, alpha=alpha, linewidth=1)
        
        if len(self.history) > 0:
            current_trace = self.history[-1]
            for i in range(3):
                self.ax.plot(range(self.width), current_trace, color=self.phosphor_green, alpha=0.2-i*0.05, linewidth=6+i*2)
            self.ax.plot(range(self.width), current_trace, color=self.phosphor_green, alpha=1.0, linewidth=2)
        
        self.ax.text(0.02, 0.95, 'CURVATURE OSCILLOSCOPE', transform=self.ax.transAxes, color=self.phosphor_green,
                     fontsize=12, fontweight='bold', family='monospace')
        
        if len(self.history) > 0:
            current_data = self.history[-1]
            peak_val = np.max(np.abs(current_data))
            rms_val = np.sqrt(np.mean(current_data**2))
            self.ax.text(0.02, 0.05, f'PEAK: {peak_val:.3f}', transform=self.ax.transAxes, color=self.phosphor_green,
                         fontsize=10, family='monospace')
            self.ax.text(0.02, 0.12, f'RMS:  {rms_val:.3f}', transform=self.ax.transAxes, color=self.phosphor_green,
                         fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.canvas.draw()
        
        try:
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height()
            scope_img = buf.reshape(h, w, 4)[:, :, :3]
        except AttributeError:
            try:
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                scope_img = buf.reshape(h, w, 3)
            except AttributeError:
                buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                scope_img = buf.reshape(h, w, 4)[:, :, 1:4]
        
        scope_img = self.add_scanlines(scope_img)
        return scope_img

    def add_scanlines(self, img):
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

# Main execution with mid-graph sharp signal detection
cap = cv2.VideoCapture(0)
oscilloscope = OscilloscopeDisplay(width=800, height=400, history_length=20)
smoother = CurvatureSmoother(maxlen=5)

# Sharp signal detection parameters - focused on mid-graph
SHARP_THRESHOLD = 0.8
MIN_PROMINENCE = 0.3
MID_RANGE_PERCENT = 0.4  # Focus on middle 40% of the graph
frame_count = 0

print("Starting Mid-Graph Sharp Signal Detection...")
print(f"Monitoring middle {MID_RANGE_PERCENT*100}% of oscilloscope trace")
print("Press ESC to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    display_frame = frame.copy()
    knot_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area > 50 and circularity > 0.7:
            knot_count += 1
            if len(cnt) >= 10:
                curv = curvature(cnt, k=15)
                curv_smooth = moving_average(curv, window_size=9)
                smoother.add(curv_smooth)
                curv_time_smooth = smoother.get_smoothed()
                
                # Process curvature data for oscilloscope
                processed_curv = abs(-1/curv_time_smooth)
                oscilloscope.add_waveform(processed_curv)
                
                # Detect sharp signals in mid-graph region
                mid_sharp_signals = detect_sharp_signals_mid_graph(
                    processed_curv, 
                    threshold=SHARP_THRESHOLD, 
                    min_prominence=MIN_PROMINENCE,
                    mid_range_percent=MID_RANGE_PERCENT
                )
                
                # Print mid-graph sharp signal detection results
                if mid_sharp_signals["count"] > 0:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Frame {frame_count}: MID-GRAPH SHARP SIGNALS!")
                    print(f"  └─ Mid-range: indices {mid_sharp_signals['mid_range'][0]}-{mid_sharp_signals['mid_range'][1]} ({mid_sharp_signals['mid_range_percent']:.1f}% of graph)")
                    print(f"  └─ Signals found: {mid_sharp_signals['count']}")
                    print(f"  └─ Max value: {mid_sharp_signals['max_value']:.3f}")
                    print(f"  └─ Avg value: {mid_sharp_signals['avg_value']:.3f}")
                    print(f"  └─ Positions: {mid_sharp_signals['indices']}")
                    print(f"  └─ Values: {[f'{v:.3f}' for v in mid_sharp_signals['values']]}")
                    print("-" * 50)
                
                # Visual feedback on detected frame
                cv2.drawContours(display_frame, [cnt], -1, (0, 0, 255), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)
                    
                    # Add mid-graph signal indicator
                    if mid_sharp_signals["count"] > 0:
                        cv2.putText(display_frame, f'MID-SHARP: {mid_sharp_signals["count"]}',
                                    (cx-40, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    
                    cv2.putText(display_frame, f'Knot: Area={area:.0f}, Circ={circularity:.2f}',
                                (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    cv2.putText(display_frame, f'Knots detected: {knot_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    scope_display = oscilloscope.render()
    cv2.imshow('Green Knot Detection', display_frame)
    cv2.imshow('Curvature Oscilloscope', scope_display)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
print("=" * 60)
print("Mid-graph oscilloscope session ended.")
