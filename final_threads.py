import cv2
import torch
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import threading
from collections import deque
import time

# Initialize the Picamera2 object
picam2 = Picamera2()

# Configure the camera and start it
camera_config = picam2.create_still_configuration(main={"size": (320, 240)})
picam2.configure(camera_config)
picam2.start()

# Set up GPIO for LED
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.HIGH)

# Glow the LED for 3 seconds on startup
time.sleep(3)
GPIO.output(LED_PIN, GPIO.LOW)

# Load the YOLO model (consider using a smaller model like YOLOv7-tiny)
model = torch.hub.load('yolov7', 'custom', path_or_model='yolov7-tiny.pt', source='local')
model.half()  # Use half precision

frame_queue = deque(maxlen=1)  # Limit queue size to avoid excessive memory usage
frame_lock = threading.Lock()

def capture_frames():
    while True:
        frame = picam2.capture_array()
        with frame_lock:
            if len(frame_queue) < frame_queue.maxlen:
                frame_queue.append(frame)

def process_frames():
    while True:
        frame = None
        with frame_lock:
            if frame_queue:
                frame = frame_queue.popleft()
        if frame is not None:
            start_time = time.time()
            
            # Convert frame to half precision
            frame = frame.astype(np.float16)
            
            # Process the frame using YOLO model
            results = model(frame, size=320)
            
            pothole_detected = False
            for detection in results.xyxy[0]:
                xmin, ymin, xmax, ymax, conf, cls = detection.cpu().numpy()
                label = results.names[int(cls)]
                score = conf
                if label == "pothole" and score > 0.5:
                    pothole_detected = True
            
            # Control the LED based on detection
            if pothole_detected:
                GPIO.output(LED_PIN, GPIO.HIGH)
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
            
            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.2f} seconds")

# Create and start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
capture_thread.start()
process_thread.start()

try:
    capture_thread.join()
    process_thread.join()
finally:
    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()
    GPIO.cleanup()
