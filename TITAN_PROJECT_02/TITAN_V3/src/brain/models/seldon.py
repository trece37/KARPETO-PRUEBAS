"""
Seldon Crisis Monitor V2
Persistence: Enabled via Joblib
Logic: Isolation Forest (Anomaly Detection)
"""
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest

class SeldonCrisisMonitor:
    def __init__(self, model_path="models/seldon_core_v2.pkl", contamination=0.01):
        self.model_path = model_path
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        self.is_fitted = False
        self.is_anomaly = False
        self.load_brain()

    def train_baseline(self, crisis_files):
        """Entrena Seldon con archivos de crisis históricos."""
        print(f"🛡️ SELDON: Entrenando con datos históricos...")
        data_vectors = []
        for f in crisis_files:
            try:
                df = pd.read_csv(f)
                df['ret'] = df['close'].pct_change()
                df['vol'] = df['ret'].rolling(20).std()
                clean = df[['ret', 'vol']].dropna()
                data_vectors.append(clean.values)
            except Exception as e:
                print(f"⚠️ Error reading {f}: {e}")

        if data_vectors:
            X = np.concatenate(data_vectors, axis=0)
            self.model.fit(X)
            self.is_fitted = True
            self.save_brain()
            print(f"✅ SELDON: Entrenado y Guardado en {self.model_path}")

    def check(self, current_return, current_volatility):
        """Consulta en vivo."""
        if not self.is_fitted:
            return False # Fail-open if not trained
            
        X_now = np.array([[current_return, current_volatility]])
        pred = self.model.predict(X_now)[0]
        
        if pred == -1:
            self.is_anomaly = True
            print(f"🚨 SELDON VETO: Anomalía detectada (Ret: {current_return:.4f}, Vol: {current_volatility:.4f})")
            return True
        
        self.is_anomaly = False
        return False

    def save_brain(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_brain(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
                print(f"🧠 SELDON: Memoria persistente cargada.")
            except:
                print("⚠️ Seldon memory corrupt or incompatible.")
        else:
            print("ℹ️ Seldon: No memory found. Waiting for training.")
