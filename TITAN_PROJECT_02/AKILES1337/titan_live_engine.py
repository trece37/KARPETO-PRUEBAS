import os
import sys
import pandas as pd
import numpy as np
import joblib
import zmq
import json
import traceback
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import SpatialDropout1D, Layer

# --------------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 00_FACTORY/TITAN_V3
MODEL_PATH = os.path.join(PROJECT_ROOT, 'output', 'v5_DEEPSEEK', 'titan_v5_deepseek.keras')
SCALER_M5_PATH = os.path.join(PROJECT_ROOT, 'output', 'v5_DEEPSEEK', 'scaler_m5.pkl')
SCALER_D1_PATH = os.path.join(PROJECT_ROOT, 'output', 'v5_DEEPSEEK', 'scaler_d1.pkl')

ZMQ_PORT = 5555
BUFFER_SIZE = 1000 # Keep last 1000 M5 candles for context calculation
SEQ_LEN = 60

# --------------------------------------------------------------------------------
# CUSTOM LAYERS (Needed for Loading Model)
# --------------------------------------------------------------------------------
# If SpatialDropout1D fails to load natively, we might need a custom scope, 
# but usually it's standard in tf.keras.layers.

# --------------------------------------------------------------------------------
# HELPER FUNCTIONS (REPLICATING TRAINING LOGIC EXACTLY)
# --------------------------------------------------------------------------------
def higuchi_fd(x, kmax=5):
    """Replicating Phase 1 Higuchi Logic"""
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.array(range(1, kmax + 1))
    y_reg = np.empty(kmax)
    
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            indices = np.arange(m, n_times, k)
            if indices.size < 2:
                lm[m] = 0
                continue
            diffs = np.abs(x[indices[1:]] - x[indices[:-1]])
            Lmk = (np.sum(diffs) * (n_times - 1) / (indices.size * k)) / k
            lm[m] = Lmk
        lk[k - 1] = np.mean(lm)
        y_reg[k - 1] = np.log(lk[k - 1] if lk[k-1] > 0 else 1e-10)
        
    x_reg = np.log(1.0 / x_reg)
    slope, intercept = np.polyfit(x_reg, y_reg, 1)
    return slope

from scipy.stats import entropy
def get_entropy(x):
    try: return entropy(pd.cut(x, 10).value_counts(normalize=True), base=2)
    except: return 0.0

# --------------------------------------------------------------------------------
# INFERENCE ENGINE CLASS
# --------------------------------------------------------------------------------
class TitanV5Engine:
    def __init__(self):
        print("🦅 INITIALIZING TITAN V5 ENGINE (HUMAN MACHINE MODE)...")
        self.load_artifacts()
        self.buffer_m5 = pd.DataFrame()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{ZMQ_PORT}")
        print(f"📡 ZMQ Listening on Port {ZMQ_PORT}")

    def load_artifacts(self):
        try:
            print(f"📂 Loading Model: {MODEL_PATH}")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"📂 Loading Scalers...")
            self.scaler_m5 = joblib.load(SCALER_M5_PATH)
            self.scaler_d1 = joblib.load(SCALER_D1_PATH)
            print("✅ Artifacts Loaded Successfully.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR LOADING ARTIFACTS: {e}")
            sys.exit(1)

    def update_candle(self, candle_data):
        """
        Ingests a new candle from MT5.
        candle_data: dict with timestamp, open, high, low, close, tick_volume
        """
        # 1. Convert timestamp to datetime
        ts = pd.to_datetime(candle_data['time'], unit='s')
        
        # 2. Append to Buffer
        new_row = pd.DataFrame([{
            'time': ts,
            'open': candle_data['open'],
            'high': candle_data['high'],
            'low': candle_data['low'],
            'close': candle_data['close'],
            'tick_volume': candle_data['tick_volume']
        }])
        new_row.set_index('time', inplace=True)
        
        self.buffer_m5 = pd.concat([self.buffer_m5, new_row])
        self.buffer_m5.sort_index(inplace=True)
        
        # Keep buffer manageable but large enough for H4/D1 calcs
        if len(self.buffer_m5) > BUFFER_SIZE:
            self.buffer_m5 = self.buffer_m5.iloc[-BUFFER_SIZE:]

    def calculate_features(self):
        """
        Replicates train_titan_v5_MULTITEMPORAL.py feature engineering ON THE FLY.
        """
        df = self.buffer_m5.copy()
        
        if len(df) < 100: # Warmup
            return None, None
            
        # --- BASE FEATURES ---
        df['time_idx'] = df.index.hour * 60 + df.index.minute
        vol_profile = df.groupby('time_idx')['tick_volume'].transform('median')
        df['rvol'] = df['tick_volume'] / (vol_profile + 1e-5)
        df['rvol'] = df['rvol'].clip(0, 5.0)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(window=12).std()
        
        # --- CAUSALITY FEATURES ---
        df['returns_raw'] = df['close'].pct_change().fillna(0)
        df['entropy'] = df['returns_raw'].rolling(20).apply(get_entropy, raw=False)
        df['entropy_z'] = (df['entropy'] - df['entropy'].rolling(252).mean()) / (df['entropy'].rolling(252).std() + 1e-6)
        
        # Optimized: Only calc HFD for relevant rows (last SEQ_LEN) to save time? 
        # For now, just calc for last 100 to be safe
        df['fractal_higuchi'] = df['close'].rolling(60).apply(lambda x: higuchi_fd(x.values, 5) if len(x)==60 else 1.5, raw=False)
        
        price_diff = df['close'].diff()
        tick_dir = np.sign(price_diff).replace(0, method='ffill').fillna(0)
        df['nofi'] = (tick_dir * df['tick_volume']) / (df['tick_volume'] + 1e-6)
        
        # --- MULTITEMPORAL CONTEXT (The Critical Part) ---
        def create_context(df, period, ma_window):
            resampled = df['close'].resample(period).last().dropna()
            ma = resampled.rolling(window=ma_window).mean()
            # Reindex & Shift to avoid lookahead
            return ma.reindex(df.index, method='ffill').shift(1)

        df['trend_h1'] = create_context(df, '1H', 20)
        df['trend_h4'] = create_context(df, '4H', 10)
        df['trend_d1'] = create_context(df, '1D', 5) # Might be NaN if buffer too short
        
        df['pos_vs_h1'] = df['close'] / (df['trend_h1'] + 1e-6)
        df['pos_vs_h4'] = df['close'] / (df['trend_h4'] + 1e-6)
        
        df_h4_atr = df['close'].resample('4H').apply(lambda x: (x.max() - x.min())).dropna()
        df['atr_h4'] = df_h4_atr.reindex(df.index, method='ffill').shift(1)
        df['volatility_norm'] = df['atr_h4'] / (df['trend_d1'].rolling(100).mean() + 1e-6)
        
        # Fill NaNs
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        # --- PREPARE TENSORS ---
        # Features M5: ['log_ret', 'rvol', 'volatility', 'entropy_z', 'fractal_higuchi', 'nofi', 'pos_vs_h1', 'pos_vs_h4', 'volatility_norm']
        feature_cols_m5 = ['log_ret', 'rvol', 'volatility', 'entropy_z', 'fractal_higuchi', 'nofi', 'pos_vs_h1', 'pos_vs_h4', 'volatility_norm']
        # Features D1: ['trend_d1', 'volatility_norm', 'atr_h4']
        feature_cols_d1 = ['trend_d1', 'volatility_norm', 'atr_h4']
        
        # Get last sequence
        last_seq_m5 = df[feature_cols_m5].iloc[-SEQ_LEN:]
        last_step_d1 = df[feature_cols_d1].iloc[-1:]
        
        if len(last_seq_m5) < SEQ_LEN:
            return None, None
            
        # Scaling
        # Note: RobustScaler is stateful, but here we use the fitted scaler from training.
        # Ideally, we should perform Winsorization similar to training using the scaler parameters?
        # Winsorization clipping logic needs to be applied using hardcoded limits or saved stats.
        # For simplicity in V1, we just apply the scaler transform.
        # TODO: Implement strict winsorization based on training set stats if anomalies occur.
        
        seq_m5_scaled = self.scaler_m5.transform(last_seq_m5)
        step_d1_scaled = self.scaler_d1.transform(last_step_d1)
        
        X_m5 = np.array([seq_m5_scaled])
        X_d1 = np.array(step_d1_scaled)
        
        return X_m5, X_d1

    def run(self):
        print("🟢 ENGINE READY. Waiting for candles...")
        while True:
            try:
                # 1. Wait for Request
                message = self.socket.recv()
                data = json.loads(message)
                
                # Check message type
                if data.get('type') == 'PING':
                    self.socket.send_string("PONG")
                    continue
                
                if data.get('type') == 'CANDLE':
                    # 2. Update Buffer
                    self.update_candle(data['payload'])
                    
                    # 3. Calculate Features
                    X_m5, X_d1 = self.calculate_features()
                    
                    if X_m5 is None:
                        # Not enough data yet
                        response = {"status": "WARMUP", "regime": -1, "prob": 0.0}
                    else:
                        # 4. Predict
                        # Returns [Range_Prob, Bull_Prob, Bear_Prob]
                        preds = self.model.predict([X_m5, X_d1], verbose=0)[0]
                        regime = int(np.argmax(preds))
                        confidence = float(preds[regime])
                        
                        response = {
                            "status": "OK",
                            "regime": regime, # 0=Range, 1=Bull, 2=Bear
                            "prob": confidence,
                            "vector": preds.tolist()
                        }
                        
                        regime_names = ["RANGO", "ALCISTA", "BAJISTA"]
                        print(f"🔮 PREDICTION: {regime_names[regime]} ({confidence:.1%})")
                    
                    self.socket.send_string(json.dumps(response))
                    
            except Exception as e:
                print(f"❌ ERROR in Loop: {e}")
                traceback.print_exc()
                self.socket.send_string(json.dumps({"status": "ERROR", "msg": str(e)}))

if __name__ == "__main__":
    engine = TitanV5Engine()
    engine.run()
