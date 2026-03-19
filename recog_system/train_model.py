import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
CSV_FILENAME = 'dataset_gestures.csv'
MODEL_FILENAME = 'gesture_classifier.joblib'
# --------------------

def load_and_preprocess_data(filename):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # Extract labels (target) and features (landmarks)
    y = df['target']
    X = df.drop('target', axis=1)
    
    # Normalization relative to the wrist (x0, y0, z0)
    # Store wrist values first to avoid using zeroed values after i=0
    wrist_x = X['x0'].copy()
    wrist_y = X['y0'].copy()
    wrist_z = X['z0'].copy()
    
    print("Normalizing landmarks (relative to the wrist)...")
    for i in range(21):
        X[f'x{i}'] = X[f'x{i}'] - wrist_x
        X[f'y{i}'] = X[f'y{i}'] - wrist_y
        X[f'z{i}'] = X[f'z{i}'] - wrist_z
    
    return X, y

def train():
    try:
        X, y = load_and_preprocess_data(CSV_FILENAME)
    except FileNotFoundError:
        print(f"Error: The file {CSV_FILENAME} was not found. Collect data first!")
        return

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("Classes found:", y.unique())

    # Create the model (Random Forest)
    print("\nTraining the RandomForest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on test set: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    print(f"Saving model to: {MODEL_FILENAME}")
    joblib.dump(clf, MODEL_FILENAME)
    print("Training successfully completed!")

if __name__ == "__main__":
    train()
