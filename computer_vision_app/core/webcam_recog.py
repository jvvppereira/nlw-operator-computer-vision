import cv2
import mediapipe as mp
import time
import os
import sys

# Adiciona o diretório 'core' ao sys.path se estiver rodando como script principal
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from config import check_models
    from model_loader import load_custom_models, get_mediapipe_options
    from processor import process_frame
except (ImportError, ValueError):
    # Caso rodado como módulo
    from .config import check_models
    from .model_loader import load_custom_models, get_mediapipe_options
    from .processor import process_frame

def main():
    if not check_models():
        return

    # Carrega o modelo customizado e o encoder de labels
    clf, label_encoder = load_custom_models()

    # Inicializa o modelo do MediaPipe Tasks
    options = get_mediapipe_options()
    GestureRecognizer = mp.tasks.vision.GestureRecognizer

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando reconhecimento CUSTOMIZADO... Pressione 'q' para sair.")

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Inverte o frame (espelhamento)
            frame = cv2.flip(frame, 1)
            
            # Timestamp necessário para o modo VIDEO do MediaPipe
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # Chama a função de processamento (imagem input -> imagem output)
            frame_processed = process_frame(
                frame, 
                recognizer, 
                clf, 
                label_encoder, 
                timestamp_ms
            )

            # Exibe o resultado
            cv2.imshow('Custom Gesture Recognition', frame_processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
