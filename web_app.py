from flask import Flask, render_template, Response
import cv2
import threading
import time
from main import DigitalUmpire

from flask import jsonify
app = Flask(__name__)

# Global variables to safely store the most recent AI-processed frames
global_umpire = None
latest_frame1 = None
latest_frame2 = None
lock = threading.Lock()

def umpire_thread():
    """Background thread that runs the AI tracker so the web server stays fast."""
    global latest_frame1, latest_frame2
    print("Initializing AI Umpire for Web...")
    
    # Intialize the same way as main.py
    global global_umpire
    umpire = DigitalUmpire()
    global_umpire = umpire
    umpire.stream.start()
    time.sleep(2.0)
    print("AI Background Umpire is active and tracking...")
    
    try:
        while True:
            f1, f2 = umpire.process_frame()
            if f1 is not None and f2 is not None:
                with lock:
                    latest_frame1 = f1
                    latest_frame2 = f2
            # Add a microscopic sleep to prevent 100% CPU lock on Windows threading
            time.sleep(0.01)
    finally:
        umpire.stream.stop()

@app.route("/")
def index():
    return render_template("index.html")

def generate_frames(camera_id):
    """Generator function to stream JPEG frames continuously to the browser."""
    global latest_frame1, latest_frame2
    while True:
        frame = None
        with lock:
            if camera_id == 1 and latest_frame1 is not None:
                frame = latest_frame1.copy()
            elif camera_id == 2 and latest_frame2 is not None:
                frame = latest_frame2.copy()
        
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Encode OpenCV frame to standard generic JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Yield the standard multipart header required for live MJPEG video feeds in HTML
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed_1")
def video_feed_1():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed_2")
def video_feed_2():
    return Response(generate_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/status")
def api_status():
    global global_umpire
    if global_umpire is not None:
        return jsonify({"status": global_umpire.latest_status})
    return jsonify({"status": "Booting up AI Models..."})

if __name__ == "__main__":
    # Start the AI in a background daemon thread
    t = threading.Thread(target=umpire_thread, daemon=True)
    t.start()
    
    # Start the Flask Web Server on port 5000
    print("\n==================================")
    print("WEB UI READY! OPEN YOUR BROWSER:")
    print("http://127.0.0.1:5000")
    print("==================================\n")
    app.run(host="127.0.0.1", port=5000, threaded=True)
