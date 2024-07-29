from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch

app = Flask(__name__)

# Load YOLOv7 model
model = torch.hub.load('yolov7', 'custom', path_or_model ='best.pt', source='local')
# model = torch.load('best.pt', map_location=torch.device('cpu')) 
model.eval()

def detect_objects(frame):
    # Perform object detection
    results = model(frame)
    # Render the detections
    rendered_frame = np.squeeze(results.render())
    return rendered_frame

def video_feed():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        rendered_frame = detect_objects(frame)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', rendered_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
