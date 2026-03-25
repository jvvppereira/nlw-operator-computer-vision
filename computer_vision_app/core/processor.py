import cv2
import numpy as np
import mediapipe as mp

def process_frame(frame, recognizer, clf, label_encoder, timestamp_ms, draw_landmarks=True):
    """
    Processes a frame (BGR), detects gestures and returns the annotated frame.
    
    Args:
        frame: Image captured by OpenCV (BGR format).
        recognizer: MediaPipe GestureRecognizer instance.
        clf: Custom classifier (joblib).
        label_encoder: Label encoder (joblib).
        timestamp_ms: Timestamp in milliseconds (required for VIDEO mode).
        draw_landmarks: Boolean to turn on/off landmark drawing.
        
    Returns:
        Annotated frame with landmarks (optional) and prediction results.
    """
    frame_annotated = frame.copy()
    
    # Prepares the image for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Extracts landmarks using MediaPipe
    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    results = []
    if recognition_result.hand_landmarks:
        mp_hands = mp.tasks.vision.HandLandmarksConnections
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_drawing_styles = mp.tasks.vision.drawing_styles

        for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            # 1. Draws landmarks on the output frame (if requested)
            if draw_landmarks:
                mp_drawing.draw_landmarks(
                    frame_annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # 2. Prepares data for the custom model
            hand_label = recognition_result.handedness[i][0].category_name
            handedness_val = 0 if hand_label == 'Left' else 1
            
            # Create a flat vector [handedness, x0, y0, z0, ..., x20, y20, z20]
            landmarks_array = [handedness_val]
            for lm in hand_landmarks:
                landmarks_array.extend([lm.x, lm.y, lm.z])
            
            # Convert to the format expected by sklearn
            features = np.array(landmarks_array).reshape(1, -1)
            
            # Custom model prediction
            prediction_idx = clf.predict(features)[0]
            prediction_prob = float(np.max(clf.predict_proba(features)))
            gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

            results.append({
                "label": hand_label,
                "gesture": gesture_name,
                "confidence": prediction_prob
            })
            
    return frame_annotated, results
