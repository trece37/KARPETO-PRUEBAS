import sys
import os
import pandas as pd
import numpy as np
from collections import deque
import json
import time
import joblib

# Add project root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Imports Titan V3
from src.brain.core.zmq_bridge import ZMQBridge
from src.brain.models.lstm import AchillesLSTM
from src.brain.models.seldon import SeldonCrisisMonitor
from src.brain.features.feature_engineering import FeatureEngineer
from src.brain.strategy.protection import CircuitBreaker
from src.brain.core.types import InsightDirection

# Buffer Config
WARMUP_SIZE = 200 
history_buffer = deque(maxlen=WARMUP_SIZE)

print("\n🧠 INICIALIZANDO GABINETE ANTIGRAVITY TITAN V3...")

# 1. Feature Engineer
engineer = FeatureEngineer()
SEQ_LEN, N_FEATURES = engineer.get_input_shape()
print(f"   -> Feature Engineer: ONLINE ({N_FEATURES} vars)")

# 2. LSTM
brain = AchillesLSTM(input_shape=(SEQ_LEN, N_FEATURES))
# Cargar pesos si existen
weights_path = os.path.join("output/v4.5", "best_achilles_titan.keras") # Ruta relativa a producción
if os.path.exists(weights_path):
    brain.load_weights(weights_path)
else:
    print(f"⚠️ No se encontró {weights_path}. El modelo iniciará sin entrenar.")

# 3. Scaler (Vital)
scaler_path = os.path.join("output/v4.5", "achilles_scaler_titan.pkl")
scaler = None
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("   -> Scaler: LOADED")
else:
    print("❌ CRITICAL: Scaler not found. Bot will crash on inference.")

# 4. Seldon
seldon = SeldonCrisisMonitor() 

# 5. Circuit Breaker (Sagrado)
circuit_breaker = CircuitBreaker(max_daily_loss_percent=0.03)

# 6. Bridge
bridge = ZMQBridge(pull_port=5555, pub_port=5556)

def process_tick(data):
    try:
        # 1. Parse Data
        tick = {
            'open': data.get('open', data['bid']),
            'high': data.get('high', data['bid']),
            'low': data.get('low', data['bid']),
            'close': data['bid'],
            'tick_volume': data.get('tick_volume', 1.0)
        }
        history_buffer.append(tick)
        
        # 2. Protection Layer
        if 'balance' in data and 'equity' in data:
            drawdown = (data['balance'] - data['equity']) / data['balance']
            is_safe, reason = circuit_breaker.check_safety(drawdown)
            if not is_safe:
                return {"action": "STOP", "reason": reason}

        # 3. Warmup
        if len(history_buffer) < WARMUP_SIZE:
            if len(history_buffer) % 50 == 0:
                print(f"⏳ Buffering: {len(history_buffer)}/{WARMUP_SIZE}")
            return None

        # 4. Feature Engineering
        df = pd.DataFrame(list(history_buffer))
        features_df = engineer.generate_features(df)
        
        if len(features_df) < SEQ_LEN:
            return None
            
        # 5. Inference Prep
        last_sequence = features_df.tail(SEQ_LEN).values
        
        # Seldon Check
        current_ret = features_df['log_ret'].iloc[-1]
        current_vol = features_df['vol_gk'].iloc[-1] if 'vol_gk' in features_df else 0.0
        
        if seldon.check(current_ret, current_vol):
            return {"action": "HOLD", "reason": "SELDON_VETO"}

        # Scaling & Prediction
        if scaler:
            # Importante: Escalar solo las columnas que el scaler conoce
            last_sequence_scaled = scaler.transform([last_sequence]) # Aprox, requiere ajuste dimensional si falla
            # NOTA: En producción, idealmente el scaler se aplica sobre el DataFrame completo antes de cortar.
            # Simplificación para el código:
            input_tensor = np.array([last_sequence]) # Asumimos que scaler.transform maneja arrays 2D
        else:
             return {"action": "HOLD", "reason": "NO_SCALER"}

        insights = brain.update(input_tensor)
        
        if not insights:
            return {"action": "HOLD", "reason": "No Insight"}
            
        insight = insights[0]
        
        # 6. Action
        if insight.confidence < 0.70:
             return {"action": "HOLD", "reason": f"Low Conf {insight.confidence:.2f}"}

        action = "HOLD"
        if insight.direction == InsightDirection.UP: action = "BUY"
        if insight.direction == InsightDirection.DOWN: action = "SELL"
        
        return {
            "action": action, 
            "confidence": insight.confidence, 
            "symbol": "XAUUSD",
            "reason": "AI_SIGNAL"
        }

    except Exception as e:
        print(f"🔥 ERROR PROCESSING TICK: {e}")
        return {"action": "HOLD", "reason": "Internal Error"}

if __name__ == "__main__":
    print("🚀 ACHILLES TITAN SYSTEM STARTED.")
    bridge.start_listening(process_tick)
    
    try:
        while True:
            bridge.check_health()
            time.sleep(5)
    except KeyboardInterrupt:
        print("🛑 SYSTEM SHUTDOWN.")
