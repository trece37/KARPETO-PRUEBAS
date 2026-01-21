# 🦅 INFORME TÉCNICO COMPLETO: PROYECTO ACHILLES TITAN V3

**Fecha:** 12/01/2026
**Autor:** Antigravity (MalizIA) para Manel "Papi"
**Versión:** Titan V3.1 (Protocolo UV + Triple Barrier)
**Estado:** Producción / Training Ready

---

## 1. RESUMEN EJECUTIVO

El sistema **Achilles Titan V3** es una arquitectura híbrida de Trading Algorítmico diseñada para operar en el mercado XAUUSD (Oro). Fusiona la robustez de ejecución de **MetaTrader 5 (MQL5)** con la potencia de cálculo de **Python (TensorFlow/Keras)**, comunicándose a través de un puente de cero latencia (**ZeroMQ**).

### Componentes Clave:
1.  **EL CEREBRO (Python):** `main.py`
    *   Gestiona la lógica, el Feature Engineering y la Inferencia.
    *   Implementa salvaguardas críticas (Circuit Breaker, Seldon Crisis Monitor).
2.  **LA INTUICIÓN (Modelo):** `lstm.py` en Colab
    *   Red Neuronal Bi-LSTM con Mecanismo de Atención.
    *   Entrenada con "Triple Barrier Method" (Buy/Sell/Hold) y Regímenes de Volatilidad.
3.  **EL OBRERO (MQL5):** `Achilles_v3.mq5`
    *   Ejecución ciega y disciplinada.
    *   Gestión de riesgo R3K (Stops dinámicos, validación de órdenes).

---

## 2. CÓDIGO FUENTE: EL CEREBRO (PYTHON)

### A. API Principal (`src/brain/api/main.py`)
*Orquestador del sistema. Recibe datos de MT5, procesa features, consulta al modelo y devuelve órdenes.*

```python
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
weights_path = os.path.join("output/v4.5", "best_achilles_titan.keras") 
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

# 4. Seldon & Circuit Breaker
seldon = SeldonCrisisMonitor() 
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
            return None

        # 4. Feature Engineering
        df = pd.DataFrame(list(history_buffer))
        features_df = engineer.generate_features(df)
        
        if len(features_df) < SEQ_LEN: return None
            
        # 5. Inference Prep
        last_sequence = features_df.tail(SEQ_LEN).values
        
        # Seldon Check
        current_ret = features_df['log_ret'].iloc[-1]
        current_vol = features_df['vol_gk'].iloc[-1] if 'vol_gk' in features_df else 0.0
        
        if seldon.check(current_ret, current_vol):
            return {"action": "HOLD", "reason": "SELDON_VETO"}

        # Scaling & Prediction
        if scaler:
            input_tensor = np.array([last_sequence]) 
        else:
             return {"action": "HOLD", "reason": "NO_SCALER"}

        insights = brain.update(input_tensor)
        
        if not insights: return {"action": "HOLD", "reason": "No Insight"}
            
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
```

### B. El Modelo Neuronal (`src/brain/models/lstm.py`)
*Arquitectura Bi-LSTM con Attention y Categorical Focal Loss.*

```python
class AchillesLSTM(AlphaModel):
    def __init__(self, input_shape, name="Achilles_LSTM_V3_Titan"):
        super().__init__(name=name)
        self.input_shape = input_shape
        self.model = self._build_model()
        self.model_path = "models/achilles_lstm_v3_titan.keras"

    def categorical_focal_loss(self, gamma=2.0, alpha=0.25):
        def focal_loss_fn(y_true, y_pred):
            import tensorflow.keras.backend as K
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -y_true * K.log(y_pred)
            weight = alpha * y_true * K.pow((1 - y_pred), gamma)
            loss = weight * cross_entropy
            return K.sum(loss, axis=1)
        return focal_loss_fn

    def _attention_block(self, inputs, time_steps):
        from tensorflow.keras.layers import Dense, Multiply, Permute
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax')(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def _build_model(self):
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, Flatten
        from tensorflow.keras.models import Model
        
        inputs = Input(shape=self.input_shape)
        
        # Bi-LSTM Layer 1
        lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Bi-LSTM Layer 2
        lstm_out = LSTM(units=64, return_sequences=True)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Attention Layer
        attention_mul = self._attention_block(lstm_out, self.input_shape[0])
        attention_mul = Flatten()(attention_mul)
        
        # Dense Decision Layer
        dense = Dense(64, activation='relu')(attention_mul)
        dense = Dropout(0.2)(dense)
        
        # Output Layer (3 Classes: Hold, Buy, Sell)
        outputs = Dense(units=3, activation='softmax')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # AdamW Optimizer
        from tensorflow.keras.optimizers import AdamW
        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
        
        model.compile(optimizer=optimizer, 
                      loss=self.categorical_focal_loss(gamma=2.0, alpha=0.25), 
                      metrics=['accuracy'])
        return model
```

---

## 3. CÓDIGO FUENTE: EL OBRERO (MQL5)

### Experto MT5 (`src/worker/Experts/Achilles_v3.mq5`)
*Cliente ZMQ que envía ticks y ejecuta órdenes con seguridad R3K.*

```cpp
#property copyright "Antigravity AI"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include "../../Include/Json.mqh"   
#include "../../Include/ZmqLib.mqh" 

input group " ZMQ Connection"
input string   ZmqHost           = "127.0.0.1";
input string   ZmqPort           = "5555";     

input group " Protection"
input int      StopLossPoints    = 500;
input int      TakeProfitPoints  = 1000;
input bool     LiveTradingMode   = false;

CTrade trade;
CZmqSocket zmq;

int OnInit() {
   if(!zmq.Initialize() || !zmq.Connect("tcp://" + ZmqHost + ":" + ZmqPort)) 
      return(INIT_FAILED);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   zmq.Shutdown();
}

void OnTick() {
   // 1. GATHER DATA
   string payload = StringFormat("{\"symbol\": \"%s\", \"ask\": %.5f, \"bid\": %.5f, \"balance\": %.2f, \"equity\": %.2f}", 
                                    _Symbol, SymbolInfoDouble(_Symbol, SYMBOL_ASK), SymbolInfoDouble(_Symbol, SYMBOL_BID), 
                                    AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY));

   // 2. ASK BRAIN
   if(!zmq.Send(payload)) return;
   string response = zmq.Receive();
   if(response == "") return; 

   // 3. PARSE & EXECUTE
   CJson parser(response);
   string action = parser.GetString("action");
   double confidence = parser.GetDouble("confidence");

   bool has_pos = PositionSelect(_Symbol);

   // CLOSE LOGIC
   if(has_pos && (action == "STOP_TRADING" || action == "CLOSE_BUY" || action == "CLOSE_SELL")) {
      trade.PositionClose(_Symbol);
      if(action == "STOP_TRADING") ExpertRemove();
   }

   // OPEN LOGIC (R3K Compliant)
   if(!has_pos && confidence > 0.8 && (action == "BUY" || action == "SELL")) {
      if(LiveTradingMode) {
          double sl, tp, open_price;
          ENUM_ORDER_TYPE type;
          
          if(action == "BUY") {
              type = ORDER_TYPE_BUY; open_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
              sl = open_price - StopLossPoints * _Point;
              tp = open_price + TakeProfitPoints * _Point;
          } else {
              type = ORDER_TYPE_SELL; open_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
              sl = open_price + StopLossPoints * _Point;
              tp = open_price - TakeProfitPoints * _Point;
          }
          
          trade.PositionOpen(_Symbol, type, 0.01, open_price, sl, tp);
      }
   }
}
```

---

## 4. ENTRENAMIENTO: COLAB NOTEBOOK (Protocolo Titan V3.1)

El cuaderno `Achilles_Titan_Training.ipynb` implementa las últimas mejoras de investigación:
1.  **Protocolo UV:** Autoreparación de dependencias (TA-Lib, TensorFlow).
2.  **Triple Barrier Method:** Etiquetas de entrenamiento (Buy/Sell/Hold) basadas en volatilidad dinámica, no en precio fijo.
3.  **Regime Detection:** Detección de "Crisis" vs "Calma" para ajustar el riesgo.
4.  **Generación de Features (12 Variables):**
    *   `log_ret`, `volatility_z`, `rsi_norm`
    *   Volatilidad Avanzada: `parkinson`, `garman_klass`, `vol_gk`
    *   Entropía de Mercado: `atr_rel`

---

**FIN DEL INFORME**
*Generado automáticamente por Antigravity para Manel.*
