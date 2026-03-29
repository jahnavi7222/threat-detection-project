from flask import Flask, render_template, Response, request
import cv2
import os
from threat_model import detect_threat

app = Flask(__name__)

# ✅ Base directory (backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Video paths (FIXED)
VIDEOS = {
    "Assault": os.path.join(BASE_DIR, "static", "videos", "assault.avi"),
    "Fighting": os.path.join(BASE_DIR, "static", "videos", "fighting.avi"),
    "Robbery": os.path.join(BASE_DIR, "static", "videos", "robbery.avi"),
    "Normal": os.path.join(BASE_DIR, "static", "videos", "normal.avi"),
    "Chasing": os.path.join(BASE_DIR, "static", "videos", "chasing.avi")
}

# ✅ Default video
selected_video = "Assault"


# 🎥 Frame generator
def generate_frames():
    global selected_video

    video_path = VIDEOS[selected_video]

    # 🔍 Debug check
    print("▶ Playing video:", video_path)
    print("📁 Exists:", os.path.exists(video_path))

    cap = cv2.VideoCapture(video_path)

    # ❌ If video not opening
    if not cap.isOpened():
        print("❌ ERROR: Cannot open video")
        return

    while True:
        success, frame = cap.read()

        if not success:
            print("❌ End of video or frame error")
            break

        # 🧠 AI Detection
        label = detect_threat(frame)

        # Store label for frontend
        app.config['CURRENT_LABEL'] = label

        # Convert frame → JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Send frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 🏠 Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_video

    if request.method == 'POST':
        selected_video = request.form.get('video')
        print("🔁 Selected video:", selected_video)

    return render_template(
        'index.html',
        videos=VIDEOS.keys(),
        selected=selected_video
    )


# 🎥 Video stream
@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# 📊 Get AI label (for UI)
@app.route('/get_label')
def get_label():
    return {"label": app.config.get('CURRENT_LABEL', 'Loading...')}


# 🚀 Run app
if __name__ == "__main__":
    app.run()