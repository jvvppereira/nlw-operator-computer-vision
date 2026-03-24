import cv2
import numpy as np
import mediapipe as mp

def process_frame(frame, recognizer, clf, label_encoder, timestamp_ms):
    """
    Processa um frame (BGR), detecta gestos e retorna o frame anotado.
    
    Args:
        frame: Imagem capturada pelo OpenCV (formato BGR).
        recognizer: Instância do MediaPipe GestureRecognizer.
        clf: Classificador customizado (joblib).
        label_encoder: Encoder das labels (joblib).
        timestamp_ms: Carimbo de tempo em milissegundos (necessário para VIDEO mode).
        
    Returns:
        Frame anotado com landmarks e resultados da predição.
    """
    frame_annotated = frame.copy()
    
    # Prepara a imagem para o MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Extrai landmarks usando MediaPipe
    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    if recognition_result.hand_landmarks:
        mp_hands = mp.tasks.vision.HandLandmarksConnections
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_drawing_styles = mp.tasks.vision.drawing_styles

        for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            # 1. Desenha os landmarks no frame de saída
            mp_drawing.draw_landmarks(
                frame_annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # 2. Prepara dados para o modelo customizado
            hand_label = recognition_result.handedness[i][0].category_name
            handedness_val = 0 if hand_label == 'Left' else 1
            
            # Criamos um vetor flat [handedness, x0, y0, z0, ..., x20, y20, z20]
            landmarks_array = [handedness_val]
            for lm in hand_landmarks:
                landmarks_array.extend([lm.x, lm.y, lm.z])
            
            # Converte para o formato esperado pelo sklearn
            features = np.array(landmarks_array).reshape(1, -1)
            
            # Predição do modelo customizado
            prediction_idx = clf.predict(features)[0]
            prediction_prob = np.max(clf.predict_proba(features))
            gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

            # 3. Exibe o resultado visualmente
            display_text = f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
            color = (0, 255, 0) # Verde para custom model
            cv2.putText(frame_annotated, display_text, (20, 50 + (i * 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    return frame_annotated
