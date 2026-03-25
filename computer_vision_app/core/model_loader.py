import joblib
import mediapipe as mp
try:
    from .config import MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH
except (ImportError, ValueError):
    from config import MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH

def load_custom_models():
    """Loads custom classifier and label encoder."""
    print("--- Loading custom models ---")
    clf = joblib.load(CUSTOM_MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return clf, label_encoder

def get_mediapipe_options():
    """Returns options to initialize MediaPipe Gesture Recognizer."""
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    return GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
