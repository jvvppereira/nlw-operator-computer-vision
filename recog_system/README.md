# Hand Gesture Recognition System

A two-fold real-time gesture recognition system that compares custom-trained Machine Learning models with native MediaPipe Gesture detection.

## 🚀 Technologies

- **[Python](https://www.python.org/)**: The core programming language.
- **[MediaPipe Tasks API](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)**: For hand landmark detection and native gesture recognition.
- **[OpenCV](https://opencv.org/)**: For webcam video capture and real-time visualization.
- **[Scikit-Learn](https://scikit-learn.org/)**: Used to train a Random Forest classifier for custom gestures.
- **[Joblib](https://joblib.readthedocs.io/)**: For model serialization and loading.
- **[Pandas & NumPy](https://pandas.pydata.org/)**: For data manipulation and coordinate normalization.

---

## 🛠️ Step-by-Step Workflow

### 1. Environment Setup
We use `uv` for dependency management. To synchronize your environment, run:
```bash
uv sync
```

### 2. Data Collection (`collect_data.py`)
Before training, we need to gather hand landmark data for specific gestures.
- Run the script: `python collect_data.py`
- Set `TARGET_LABEL` in the script for the gesture you want to record (e.g., `rock`, `hang_loose`, `small_heart`).
- Press **'a'** to toggle "Auto-Save" mode. It will record 21 landmarks (x, y, z) for every frame where a hand is detected.
- Data is saved into `dataset_gestures.csv`.

### 3. Model Training (`train_model.py`)
Once you have enough samples (aim for 500+ per label), run the training script:
```bash
python train_model.py
```
This script:
- Loads the CSV data.
- **Normalizes** the landmarker coordinates (making them relative to the wrist).
- Trains a **Random Forest Classifier**.
- Saves the trained model as `gesture_classifier.joblib`.

### 4. Real-Time Demonstration (`webcam_mediapipe.ipynb`)
Run the Jupyter Notebook and choose between two sections:
1. **Custom ML Recognition**: Uses your newly trained `joblib` model. It detects hands with MediaPipe, normalizes the live data, and predicts your custom labels.
2. **Native MediaPipe Recognition**: Uses the built-in `gesture_recognizer.task` to identify standard gestures (Victory, Thumbs Up, etc.).

---

## 🔬 Advanced Computer Vision Notebooks

This project also includes several specialized notebooks for different Computer Vision tasks:

### 1. Image Classification (`timm_classification.ipynb`)
- **Library**: `timm` (PyTorch Image Models).
- **Purpose**: Classify static images into various categories using pre-trained deep learning architectures.

### 2. Object Detection (`yolos_detection.ipynb`)
- **Model**: `hustvl/yolos-small`.
- **Purpose**: Detect bounding boxes and classify multiple objects within a single image.

### 3. Image Segmentation (`clipseg_segmentation.ipynb`)
- **Model**: `clipseg`.
- **Purpose**: Perform "prompt-based" segmentation (Semantic Segmentation), allowing you to extract mask regions of specific items in an image using natural language descriptions.

### 4. Gemini AI Vision (`gemini_vision.ipynb`)
- **SDK**: `google-generativeai`.
- **Purpose**: Use Google's Gemini Multimodal models to analyze images and answer complex visual questions via the API.

---

## 🎮 How to Use the Notebooks
1. Ensure your environment is ready: `uv sync`.
2. For Gemini Vision, make sure your `.env` file contains your `GEMINI_API_KEY`.
3. Open any of the `.ipynb` files and run the cells sequentially to see the models in action.
4. For the webcam demos, stay in a well-lit area and press **'q'** to close the window.
