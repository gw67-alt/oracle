import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def moving_average(data, window_size=7):
    """Smooth data with a simple moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

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
        curvature = 1/r if r != 0 else 0
        curvatures.append(curvature)
    return np.array(curvatures)

cap = cv2.VideoCapture(0)
plt.ion()
smoother = CurvatureSmoother(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    display_frame = frame.copy()
    curvature_chart = np.zeros((300, 800, 3), dtype=np.uint8)

    for cnt in contours:
        if len(cnt) < 20:
            continue
        curv = curvature(cnt, k=15)
        curv_smooth = moving_average(curv, window_size=9)
        smoother.add(curv_smooth)
        curv_time_smooth = smoother.get_smoothed()

        # Plot both the latest (smoothed) and historical average
        plt.clf()
        plt.plot(curv_smooth, label='Frame Smoothed', alpha=0.5)
        if len(curv_time_smooth) > 0:
            plt.plot(curv_time_smooth, label='Temporal Average', linewidth=2)
        plt.title('Curvature along contour')
        plt.xlabel('Contour Point Index')
        plt.ylabel('Curvature')
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig = plt.gcf()
        fig.canvas.draw()
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        plot_img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        plot_img = plot_img[:, :, [1, 2, 3]]
        curvature_chart = cv2.resize(plot_img, (800, 300))
        cv2.drawContours(display_frame, [cnt], -1, (0, 0, 255), 2)
        break

    cv2.imshow('Green Object Contour', display_frame)
    cv2.imshow('Curvature Chart', curvature_chart)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
