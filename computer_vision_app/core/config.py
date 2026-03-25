import os

# Paths to models calculated relative to this file
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CORE_DIR, "..", "models")

MP_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_recognizer.task") 
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

def check_models():
    """Checks if all necessary model files exist."""
    models = [MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH]
    missing = [p for p in models if not os.path.exists(p)]
    if missing:
        print(f"Error: Files not found: {', '.join(missing)}")
        return False
    return True
