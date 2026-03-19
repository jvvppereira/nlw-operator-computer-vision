import cv2
import mediapipe as mp
import csv
import os
import time

# --- DATASET CONFIGURATION ---
CSV_FILENAME = 'dataset_gestures.csv'
TARGET_LABEL = 'small_heart'  # Gesture name for saved samples
AUTO_SAVE = True       # If True, automatically saves whenever a hand is detected
# ------------------------------

# MediaPipe setup (Tasks API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

def save_to_csv(landmarks, label, filename):
    """Saves the 21 points (x, y, z) to a CSV."""
    data = [label]
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z])
        
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['target']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(header)
        writer.writerow(data)

def main():
    global AUTO_SAVE
    print(f"Starting data collection (AUTO_SAVE={AUTO_SAVE}) for: {TARGET_LABEL}")
    print("Press 'a' to toggle auto-save.")
    print("Press 's' to save manually.")
    print("Press 'q' to quit.")
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2  # Allowing collection from both hands if necessary
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        save_counter = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = landmarker.detect(mp_image)
            
            h, w, _ = frame.shape
            
            if result.hand_landmarks:
                # Loop through all detected hands
                for hand_landmarks in result.hand_landmarks:
                    # Save to CSV if Auto Save is on
                    if AUTO_SAVE:
                        save_to_csv(hand_landmarks, TARGET_LABEL, CSV_FILENAME)
                        save_counter += 1
                        # Visual feedback per hand (colored border indicating recording)
                        cv2.rectangle(frame, (0,0), (w,h), (0, 0, 255), 2)
                    
                    # Draw skeleton
                    for connection in HAND_CONNECTIONS:
                        start_pt = (int(hand_landmarks[connection[0]].x * w), 
                                    int(hand_landmarks[connection[0]].y * h))
                        end_pt = (int(hand_landmarks[connection[1]].x * w), 
                                  int(hand_landmarks[connection[1]].y * h))
                        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
                    
                    for landmark in hand_landmarks:
                        cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, (0, 0, 255), -1)

            # Status Interface
            status_color = (0, 0, 255) if AUTO_SAVE else (255, 255, 255)
            status_text = "AUTO-SAVING" if AUTO_SAVE else "MANUAL MODE"
            
            cv2.putText(frame, f"LABEL: {TARGET_LABEL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"STATUS: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"SAVED: {save_counter}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "'a' toggle auto | 's' manual | 'q' quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Data Collector Pro', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        save_to_csv(hand_landmarks, TARGET_LABEL, CSV_FILENAME)
                        save_counter += 1
                    print(f"Manually saved: {TARGET_LABEL}")
            elif key == ord('a'):
                AUTO_SAVE = not AUTO_SAVE
                print(f"Auto-save: {AUTO_SAVE}")
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
