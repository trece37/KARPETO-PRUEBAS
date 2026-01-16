# --------------------------------------------------------------------------------
# 🎱 ARCHIVO: CONFIG/SETTINGS.PY
# PROYECTO: ACHILLES TITAN V3
# AUTOR: ANTIGRAVITY (Bajo Orden del Comandante Manel)
# FUENTE: Variables y configuraciones bot MetaTrader.md + Dirty 100
# CLASIFICACIÓN: R3K STRICT CHECKING
# --------------------------------------------------------------------------------
import os

class TradingConfig:
    """
    CONFIGURACIÓN CENTRAL DEL SISTEMA TITAN V3.
    Mapeo estricto de variables para sincronización perfecta con MT5 (El Obrero).
    """
    
    # ==============================================================================
    # 💀 1. IDENTIDAD Y CONTROL (HEADER)
    # ==============================================================================
    BOT_NAME = "ACHILLES_TITAN_V3"
    VERSION = "3.1.0-BlackOps"
    # Fuente: Variables.md -> "MagicNumber (identificador del EA)"
    MAGIC_NUMBER = 1337666  
    
    # ==============================================================================
    # 💀 2. GESTIÓN MONETARIA Y RIESGO (R3K: RULES 81-85 Dirty 100)
    # ==============================================================================
    # Fuente: Variables.md -> "RiskPercent", "MaxDrawdown"
    RISK_PER_TRADE = 1.0        # % de la cuenta a arriesgar por trade (La Regla del 2% Max)
    MAX_DRAWDOWN_HARD = 20.0    # % de DD donde el bot se suicida (Kill Switch)
    
    # Fuente: Variables.md -> "TradingHours", "SymbolFilter"
    SYMBOL = "XAUUSD"
    TIMEFRAME = "M15"           # Temporalidad Base del Cerebro
    
    # Ratchet & Stop logic
    MIN_RISK_REWARD = 1.5       # R:R mínimo aceptable
    MAX_SPREAD_PIPS = 30        # Filtro de spread (Variables.md -> SpreadMax)
    
    # ==============================================================================
    # 💀 3. CEREBRO LSTM (TITAN ARCHITECTURE)
    # ==============================================================================
    # Hiperparámetros alineados con Dirty 100 -> "Sección C: AI & Deep Learning"
    LOOKBACK_WINDOW = 60        # 'LSTM Input' (Dirty 41)
    FORECAST_HORIZON = 1        
    
    # Arquitectura Neuronal
    LSTM_UNITS = 128
    LSTM_LAYERS = 2
    DROPOUT_RATE = 0.2          # 'Dropout' (Dirty 45)
    LEARNING_RATE = 0.001       # 'AdamW' standard
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # ==============================================================================
    # 💀 4. INGENIERÍA DE FEATURES (DIRTY 100 HACKS)
    # ==============================================================================
    # Activación de hacks específicos de la Enciclopedia
    FEATURES = {
        # --- PRECIOS & RETORNOS ---
        'USE_LOG_RETURNS': True,     # Dirty 6: "Los modelos prefieren np.log"
        
        # --- VOLATILIDAD ---
        'USE_PARKINSON_VOL': True,   # Dirty 17: "Mejor que Close"
        'PARKINSON_WINDOW': 14,
        'USE_ATR': True,             # R3K Standard
        'ATR_PERIOD': 14,
        
        # --- MOMENTUM ---
        'USE_RSI': True,
        'RSI_PERIOD': 14,
        'USE_MACD': True,
        'MACD_FAST': 12, 'MACD_SLOW': 26, 'MACD_SIGNAL': 9,
        
        # --- FILTROS DE RUIDO ---
        'USE_ADX_FILTER': True,      # Dirty 23: "Si ADX < 20, rango"
        'ADX_PERIOD': 14,
        
        # --- DATA DIRTY HACKS ---
        'USE_DAY_OF_WEEK': True,     # Dirty 9
        'USE_HOUR_OF_DAY': True,     # Dirty 10
        'USE_ZSCORE_NORM': True      # Dirty 7: Rolling Z-Score
    }
    
    # ==============================================================================
    # 💀 5. RUTAS (PATHING)
    # ==============================================================================
    # Definición dinámica para soportar Colab y Local
    BASE_PATH = os.getcwd()
    DATA_PATH = os.path.join(BASE_PATH, "data")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    LOGS_PATH = os.path.join(BASE_PATH, "logs")

