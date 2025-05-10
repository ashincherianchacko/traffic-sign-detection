!pip install ipywidgets

"""
Traffic Sign Detection using OpenCV for Google Colab
This script captures video from a webcam in Google Colab and detects potential traffic signs based on color.
"""

import cv2
import numpy as np
import os
import time
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import matplotlib.pyplot as plt
import io

# Define constants for different traffic sign colors
# These HSV ranges can be adjusted for better detection in your environment
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

def take_photo(filename='photo.jpg', quality=0.8):
    # Convert quality to a string to avoid the timeout error
    quality_str = str(quality)
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            const stopButton = document.createElement('button');
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');

            capture.textContent = 'Capture';
            stopButton.textContent = 'Stop Camera';
            div.appendChild(video);
            div.appendChild(capture);
            div.appendChild(stopButton);

            document.body.appendChild(div);

            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            video.srcObject = stream;
            await video.play();

            // Resize the canvas to match the video feed
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;
            canvas.width = videoWidth;
            canvas.height = videoHeight;

            let captureInterval = null;
            let captureFlag = false;
            let canvasShown = false;

            capture.onclick = function() {
                if (!captureFlag) {
                    captureFlag = true;
                    capture.textContent = 'Stop Capturing';
                    // Start continuous capture
                    captureInterval = setInterval(() => {
                        const context = canvas.getContext('2d');
                        context.drawImage(video, 0, 0, videoWidth, videoHeight);
                        if (!canvasShown) {
                            div.appendChild(canvas);
                            canvasShown = true;
                        }
                        canvas.toBlob(function(blob) {
                            if (blob) {
                                const reader = new FileReader();
                                reader.onload = function(e) {
                                    const base64data = e.target.result;
                                    google.colab.kernel.invokeFunction('notebook.run', [base64data], {});
                                };
                                reader.readAsDataURL(blob);
                            }
                        }, 'image/jpeg', quality);
                    }, 100); // Capture interval in ms
                } else {
                    captureFlag = false;
                    capture.textContent = 'Capture';
                    clearInterval(captureInterval);
                }
            };

            stopButton.onclick = function() {
                clearInterval(captureInterval);
                video.pause();
                stream.getTracks().forEach(track => {
                    track.stop();
                });
                div.remove();
            };
        }

        takePhoto(''' + quality_str + '''); // Pass quality as a string
    ''')
    display(js)

def js_to_image(js_reply):
    """Convert JS response to OpenCV image"""
    if not js_reply:
        return None

    # Remove the data URL prefix and decode
    b64_data = js_reply.split(',')[1]
    binary = b64decode(b64_data)

    # Convert to numpy array
    image = np.asarray(bytearray(binary), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def detect_by_color(frame):
    """Detect potential traffic signs based on their color"""
    detected_objects = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red signs (requires two ranges in HSV)
    for sign_type in RED_SIGNS:
        mask1 = cv2.inRange(hsv, sign_type['lower1'], sign_type['upper1'])
        mask2 = cv2.inRange(hsv, sign_type['lower2'], sign_type['upper2'])
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 400:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append((x, y, w, h, sign_type['name'], (0, 0, 255)))  # Red color for bbox

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
                detected_objects.append((x, y, w, h, sign_type['name'], (255, 0, 0)))  # Blue color for bbox

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
                detected_objects.append((x, y, w, h, sign_type['name'], (0, 255, 255)))  # Yellow color for bbox

    return detected_objects

def analyze_shape(roi):
    """Basic shape analysis to help identify sign type"""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "Unknown"

    # Get the largest contour
    cnt = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Count the number of vertices
    vertices = len(approx)

    # Analyze shape based on vertices
    if vertices == 3:
        return "Triangle (Warning)"
    elif vertices == 4:
        # Check if it's a square or rectangle
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

def process_and_display(frame):
    """Process the frame and detect traffic signs"""
    if frame is None:
        return

    # Create a copy of the frame for drawing
    output_frame = frame.copy()

    # Detect traffic signs based on color
    detected_objects = detect_by_color(frame)

    # Process detected objects
    for (x, y, w, h, sign_type, color) in detected_objects:
        # Extract the region of interest
        roi = frame[y:y+h, x:x+w]

        # Check if ROI is not empty
        if roi.size == 0:
            continue

        # Perform shape analysis for better classification
        shape_info = analyze_shape(roi)

        # Draw bounding box with color based on sign type
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)

        # Display label with sign type and shape info
        text = f"{sign_type} - {shape_info}"
        # Add black background for better text visibility
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(output_frame, (x, y-text_size[1]-10), (x+text_size[0], y), (0, 0, 0), -1)
        # Add text
        cv2.putText(output_frame, text, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    # Display with matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

    return detected_objects

# Register function to be called from JavaScript
from google.colab import output
output.register_callback('notebook.run', process_and_display)

def main():
    """Main function to run the traffic sign detection"""
    print("Starting Traffic Sign Detection in Google Colab...")
    print("Click 'Capture' to start the webcam feed")
    print("The detected traffic signs will be displayed below each frame")
    take_photo()

if __name__ == "__main__":
    main()
