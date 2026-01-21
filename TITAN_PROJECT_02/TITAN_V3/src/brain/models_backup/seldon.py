import numpy as np
import pandas as pd
import os
import joblib

from sklearn.covariance import EllipticEnvelope
from typing import List, Optional
from ..core.interfaces import RiskManagementModel
from ..core.types import PortfolioTarget

class SeldonCrisisMonitor(RiskManagementModel):
    def __init__(self, contamination=0.01):
        """
        Seldon Crisis Monitor V2: Multivariate Anomaly Detection (Crashes).
        Uses Elliptic Envelope (Robust Covariance) to identify outliers in
        PRICE RETURNS and VOLATILITY.
        """
        self.model = EllipticEnvelope(contamination=contamination)
        self.is_fitted = False
        self.history = [] 
        self.window_size = 100
        self.last_return = 0.0
        self.last_vol = 0.0
        self.is_anomaly = False
        # [TAG: MULTIVARIATE_VERSIONING]
        # Changed filename to v2 to prevent loading incompatible 1D models
        self.model_filename = "seldon_model_v2.joblib"

    def load_baseline(self, file_paths: List[str]):
        """
        Loads multiple CSV files, calculates Returns AND Volatility, and fits the Seldon model.
        Expects files to have 'close' or 'Close' column.
        """
        print(f"Seldon V2: Loading {len(file_paths)} historical files for multivariate baseline...")
        all_features = []
        
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"Warning: File not found {fp}")
                continue
            
            try:
                df = pd.read_csv(fp)
                df.columns = [c.lower() for c in df.columns]
                
                if 'close' not in df.columns:
                    print(f"Warning: 'close' column missing in {fp}")
                    continue
                
                # Calculate Returns
                df['ret'] = df['close'].pct_change()
                
                # Calculate Volatility (20-period Rolling Std Dev of Returns) - Proxy for Regime
                # We need a rolling window, so first 20 will be NaN
                df['vol'] = df['ret'].rolling(window=20).std()
                
                # Drop NaNs
                df_clean = df[['ret', 'vol']].dropna()
                
                if len(df_clean) > 0:
                    features = df_clean.values # [[ret, vol], ...]
                    all_features.extend(features.tolist())
                    print(f"Loaded {len(features)} points from {os.path.basename(fp)}")
                
            except Exception as e:
                print(f"Error loading {fp}: {e}")

        if not all_features:
            print("CRITICAL: Seldon could not load any data! Monitor remains unfitted.")
            return

        # Deduplication
        X = np.array(all_features)
        # Unique rows
        X_unique = np.unique(X, axis=0) # Axis 0 = rows
        
        if len(X_unique) < len(X):
            duplicates = len(X) - len(X_unique)
            print(f"⚠️ DEDUPLICATION: Removed {duplicates} duplicate vectors ({duplicates/len(X)*100:.1f}%)")
        
        self.fit(X_unique)

    def fit(self, training_data: np.ndarray):
        """
        Fits the anomaly detector on multivariate data (Returns, Volatility).
        training_data shape: (N_samples, 2)
        """
        self.model.fit(training_data)
        self.is_fitted = True
        print(f"Seldon V2 Monitor Fitted on {len(training_data)} vectors. Dimensions: {training_data.shape[1]}")
        self.save_model(self.model_filename)

    def save_model(self, filepath: str):
        try:
            joblib.dump(self.model, filepath)
            print(f"Seldon V2 Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving Seldon model: {e}")

    def load_model(self, filepath: Optional[str] = None) -> bool:
        if filepath is None:
            filepath = self.model_filename
            
        if not os.path.exists(filepath):
            return False
        try:
            self.model = joblib.load(filepath)
            
            # Check dimensions (Quick heuristics validation)
            if hasattr(self.model, 'location_'):
                dims = self.model.location_.shape[0]
                if dims != 2:
                    print(f"⚠️ Warning: Loaded model has {dims} dimensions, expected 2. Discarding.")
                    return False
            
            self.is_fitted = True
            print(f"Seldon V2 Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading Seldon model: {e}")
            return False

    def update(self, current_return: float, current_vol: float = 0.0):
        """
        Updates the monitor with the latest return AND volatility. 
        """
        self.last_return = current_return
        self.last_vol = current_vol
        
        if not self.is_fitted:
            return

        # Feature Vector: [Return, Volatility]
        # Note: If Volatility is 0 passed (e.g. from simplistic caller), 
        # it might trigger anomaly if the model expects normal vol levels.
        # Ideally caller SHOULD pass real volatility.
        
        vector = [[current_return, current_vol]]
        
        prediction = self.model.predict(vector)[0]
        
        if prediction == -1:
            self.is_anomaly = True
            # print(f"SELDON V2 ALERT: Anomaly Detected (Ret: {current_return:.4%}, Vol: {current_vol:.4%})")
        else:
            self.is_anomaly = False

    def manage_risk(self, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        If a Crisis (Anomaly) is detected, liquidate all Long positions.
        """
        if not self.is_fitted:
            return targets 

        if self.is_anomaly:
             print(f"SELDON INTERVENTION: Vetoing all trades. (Ret: {self.last_return:.4%}, Vol: {self.last_vol:.4%})")
             return [PortfolioTarget(symbol=t.symbol, quantity=0, percent=0.0) for t in targets]
        
        return targets
