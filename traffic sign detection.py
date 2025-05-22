"""
Real-Time Traffic Sign Detection using OpenCV
This script captures video from a webcam and detects potential traffic signs based on color and shape.
"""

import cv2
import numpy as np

# Define constants for different traffic sign colors
RED_SIGNS = [
    {'name': 'Stop/Prohibition', 'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
     'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])},
]

BLUE_SIGNS = [
    {'name': 'Information/Mandatory', 'lower': np.array([100, 80, 70]), 'upper': np.array([130, 255, 255])},
]

YELLOW_SIGNS = [
    {'name': 'Warning', 'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
]

def detect_by_color(frame):
    """Detect potential traffic signs based on their color"""
    detected_objects = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red signs (requires two ranges in HSV)
    for sign_type in RED_SIGNS:
        mask1 = cv2.inRange(hsv, sign_type['lower1'], sign_type['upper1'])
        mask2 = cv2.inRange(hsv, sign_type['lower2'], sign_type['upper2'])
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 400:
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append((x, y, w, h, sign_type['name'], (0, 0, 255)))

    # Detect blue signs
    for sign_type in BLUE_SIGNS:
        mask = cv2.inRange(hsv, sign_type['lower'], sign_type['upper'])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 400:
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append((x, y, w, h, sign_type['name'], (255, 0, 0)))

    # Detect yellow signs
    for sign_type in YELLOW_SIGNS:
        mask = cv2.inRange(hsv, sign_type['lower'], sign_type['upper'])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 400:
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append((x, y, w, h, sign_type['name'], (0, 255, 255)))

    return detected_objects

def analyze_shape(roi):
    """Basic shape analysis to help identify sign type"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "Unknown"

    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle (Warning)"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:
            return "Square (Information)"
        else:
            return "Rectangle (Regulatory)"
    elif vertices == 8:
        return "Octagon (Stop)"
    elif vertices > 8:
        return "Circle (Prohibition/Mandatory)"
    else:
        return "Unknown"

def main():
    """Main function to run the traffic sign detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Traffic Sign Detection...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_by_color(frame)

        for (x, y, w, h, sign_type, color) in detected_objects:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            shape_info = analyze_shape(roi)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{sign_type} - {shape_info}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0], y), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Traffic Sign Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
