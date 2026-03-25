from fasthtml.common import *
from starlette.staticfiles import StaticFiles
import sys
import os
import time
import json
import mediapipe as mp

# Add 'core' directory to sys.path to recognize internal modules
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, "core")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

from config import check_models
from model_loader import load_custom_models, get_mediapipe_options
from processor import process_frame
from utils import decode_image, encode_image

# Initialize models (globally)
if not check_models():
    print("Error: Models not found!")
    clf = label_encoder = recognizer = None
else:
    clf, label_encoder = load_custom_models()
    options = get_mediapipe_options()
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    recognizer = GestureRecognizer.create_from_options(options)

app, rt = fast_app()

# Serve static files from 'assets' folder
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@rt("/")
def get():
    # Professional HTML structure with External CSS
    return (
        Title("Hand Gesture AI Recognition"),
        Link(rel="stylesheet", href="/assets/style.css"),
        Main(
            H1("Gesture Recognition AI"),
            
            # Main Section Flex (Canvas on left, Control Panel and Match on right)
            Div(
                Canvas(id="canvas", width="640", height="480"),
                
                # Side Panel (Vertical)
                Div(
                    # Controls next to the canvas
                    Div(
                        Div(
                            Label("Quality: ", For="quality-slider"),
                            Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.1", value="0.5"),
                            Span("0.5", id="quality-value")
                        ),
                        Div(
                            Label("Draw Landmarks: ", For="draw-landmarks"),
                            Input(type="checkbox", id="draw-landmarks", checked=True),
                        ),
                        id="controls"
                    ),

                    # Match (Below controls now in the same side panel)
                    Div(
                        H2("Match Detected! ✨"),
                        Img(id="match-image", src="", width="200"),
                        id="match-container"
                    ),
                    id="side-panel"
                ),
                id="webcam-section"
            ),

            # Div for labels and FPS (now below webcam-section)
            Div(
                Div(Span("FPS: "), Span("0", id="fps-value"), id="fps-counter"),
                Div(id="labels"),
                id="header-info"
            ),
            
            Video(id="video", autoplay=True, width="640", height="480", style="display:none"),
            Script(src="/assets/script.js")
        )
    )

# Variables for FPS calculation (Backend)
fps_last_time = time.time()
fps_frame_count = 0
current_fps = 0

@app.ws("/ws")
async def ws(image: str, ws, draw_landmarks: bool = True):
    global fps_last_time, fps_frame_count, current_fps
    
    # Decodes image from frontend
    frame = decode_image(image)
    if frame is None: return

    # Processes frame (requires timestamp) using global models
    timestamp_ms = int(time.time() * 1000)
    
    labels = []
    match_gesture = None
    
    if recognizer:
        processed, labels = process_frame(frame, recognizer, clf, label_encoder, timestamp_ms, draw_landmarks=draw_landmarks)
        
        # Match Logic: 2 hands with the same gesture
        if len(labels) == 2 and labels[0]["gesture"] == labels[1]["gesture"]:
            match_gesture = labels[0]["gesture"]
    else:
        processed = frame

    # Encodes image to send to frontend
    encoded_image = encode_image(processed)
    
    # Processing FPS calculation
    fps_frame_count += 1
    now = time.time()
    if now - fps_last_time >= 1.0:
        current_fps = round(fps_frame_count / (now - fps_last_time), 1)
        fps_frame_count = 0
        fps_last_time = now

    # Sends JSON with image, labels, optional match, and FPS
    await ws.send_text(json.dumps({
        "image": encoded_image,
        "labels": labels,
        "match_gesture": match_gesture,
        "fps": current_fps
    }))

serve()
