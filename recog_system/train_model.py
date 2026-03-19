import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURAÇÃO ---
CSV_FILENAME = 'dataset_gestures.csv'
MODEL_FILENAME = 'gesture_classifier.joblib'
# --------------------

def load_and_preprocess_data(filename):
    print(f"Carregando dados de {filename}...")
    df = pd.read_csv(filename)
    
    # Extrair labels (target) e features (landmarks)
    y = df['target']
    X = df.drop('target', axis=1)
    
    # Normalização relativa ao pulso (x0, y0, z0)
    # Salvamos valores do pulso primeiro para não usarmos valor zerado após i=0
    wrist_x = X['x0'].copy()
    wrist_y = X['y0'].copy()
    wrist_z = X['z0'].copy()
    
    print("Normalizando landmarks (relativo ao pulso)...")
    for i in range(21):
        X[f'x{i}'] = X[f'x{i}'] - wrist_x
        X[f'y{i}'] = X[f'y{i}'] - wrist_y
        X[f'z{i}'] = X[f'z{i}'] - wrist_z
    
    return X, y

def train():
    try:
        X, y = load_and_preprocess_data(CSV_FILENAME)
    except FileNotFoundError:
        print(f"Erro: O arquivo {CSV_FILENAME} não foi encontrado. Colete dados primeiro!")
        return

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Amostras de treino: {len(X_train)}")
    print(f"Amostras de teste: {len(X_test)}")
    print("Classes encontradas:", y.unique())

    # Criar o modelo (Random Forest)
    print("\nTreinando o modelo RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Avaliação
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia no conjunto de teste: {accuracy:.2f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Salvar o modelo
    print(f"Salvando o modelo em: {MODEL_FILENAME}")
    joblib.dump(clf, MODEL_FILENAME)
    print("Treino concluído com sucesso!")

if __name__ == "__main__":
    train()
