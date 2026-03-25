# Hand Gesture AI Recognition 🚀

A real-time computer vision application that leverages **MediaPipe**, **FastHTML**, and **WebSockets** to perform hand gesture recognition with high performance and a premium user experience.

## ✨ Features

- **Real-time Processing**: Seamless video streaming and processing using WebSockets for ultra-low latency.
- **Custom AI Intelligence**: Combines MediaPipe's landmark detection with a custom Scikit-Learn classifier to recognize specific gestures (e.g., *rock*, *hang loose*, *small heart*).
- **Responsive Controls**:
    - **Quality Slider**: Real-time control of image compression to optimize bandwidth.
    - **Landmark Toggle**: Checkbox to enable or disable the visualization of hand connections on the screen.
- **Performance Monitoring**: Backend-measured FPS counter to track processing efficiency.
- **Interactive Match System**: Detects when the same gesture is performed with both hands, triggering a visual "Match Detected" state with gesture-specific images.
- **Elevated UI/UX**: Professional **Glassmorphism** design with a side control panel and responsive layout.

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed (3.8+ recommended).

### 2. Setup environment
If using **uv**:
```bash
uv run python app.py
```

Using **pip**:
```bash
pip install -r pyproject.toml # or install necessary dependencies
python app.py
```

*Note: The application will start a web server at `http://localhost:5001` (by default with FastHTML).*

## 📁 Project Structure

- `app.py`: Main entry point. Defines the FastHTML UI structure and WebSocket communication logic.
- `assets/`: Static files (JavaScript, CSS, Images).
    - `script.js`: Handles webcam access, WebSocket messaging, and UI updates.
    - `style.css`: Implements global styling with a modern dark theme.
- `core/`: Python processing modules.
    - `processor.py`: Orchestrates frame processing and landmark extraction.
    - `model_loader.py`: Handles loading of the gesture models.
    - `utils.py`: Image encoding/decoding utilities.
- `models/`: Trained model binaries (.task and .joblib).

## 🛠️ Technical Stack

- **Framework**: [FastHTML](https://fasthtml.answer.ai/) (FastAPI-based).
- **Computer Vision**: [MediaPipe](https://mediapipe.dev/) & OpenCV.
- **Machine Learning**: Scikit-Learn (Joblib for model persistence).
- **Frontend**: Vanilla JavaScript (Web APIs) & Modern CSS.
