# --------------------------------------------------------------------------------
# 🔬 ARCHIVO: SRC/BRAIN/FEATURES/FEATURE_ENGINEERING.PY
# PROYECTO: ACHILLES TITAN V3
# AUTOR: ANTIGRAVITY
# FUENTE: TITAN V3 Enciclopedia de Guerra Algorítmica (Dirty 100)
# CLASIFICACIÓN: DATA ALCHEMY (R6V VERIFIED)
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pandas_ta as ta
from config.settings import TradingConfig

class FeatureEngineer:
    """
    MOTOR DE INGENIERÍA DE CARACTERÍSTICAS TITAN V3.
    Implementa los 'Dirty Hacks' de la Enciclopedia para maximizar la relación Señal/Ruido.
    """
    
    def __init__(self):
        self.cfg = TradingConfig
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Tick_Volume']

    def validate_data(self, df):
        """ Dirty 5: Validación de Velas. Eliminar basura. """
        initial_len = len(df)
        # Asegurar columnas
        if not all(col in df.columns for col in self.required_columns):
            raise ValueError(f"Faltan columnas requeridas. Se necesitan: {self.required_columns}")
            
        # Filtro de integridad (High >= Low)
        df = df[df['High'] >= df['Low']].copy()
        
        # Filtro de volumen negativo (si existe)
        df = df[df['Tick_Volume'] >= 0].copy()
        
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"💀 [DIRTY HACK #5] Se purgaron {dropped} velas corruptas.")
            
        return df

    def apply_log_returns(self, df):
        """ Dirty 6: Log Returns (Simétricos y Aditivos) """
        if self.cfg.FEATURES['USE_LOG_RETURNS']:
            # Log Return = ln(Close_t / Close_t-1)
            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
            print("   ✅ [MATH] Retornos Logarítmicos calculados.")
        return df

    def apply_parkinson_volatility(self, df):
        """ Dirty 17: Volatilidad Parkinson (Usa High/Low, mas información que Close) """
        if self.cfg.FEATURES['USE_PARKINSON_VOL']:
            # Formula: sqrt(1 / (4 * ln(2)) * (ln(High/Low))^2)
            const = 1.0 / (4.0 * np.log(2.0))
            df['parkinson_vol'] = np.sqrt(const * (np.log(df['High'] / df['Low']) ** 2))
            
            # Aplicamos ventana móvil para suavizar (Tendencia de volatilidad)
            window = self.cfg.FEATURES['PARKINSON_WINDOW']
            df['parkinson_vol_ma'] = df['parkinson_vol'].rolling(window=window).mean()
            print("   ✅ [MATH] Volatilidad Parkinson (Dirty Hack #17) calculada.")
        return df

    def apply_rolling_zscore(self, df, col='Close', window=20):
        """ Dirty 7: Z-Score Rolling (Normalización Local) """
        if self.cfg.FEATURES['USE_ZSCORE_NORM']:
            # Z = (X - Mean) / Std
            roll_mean = df[col].rolling(window=window).mean()
            roll_std = df[col].rolling(window=window).std()
            
            # Evitar división por cero
            df[f'{col}_zscore'] = (df[col] - roll_mean) / (roll_std + 1e-8)
            print(f"   ✅ [MATH] Z-Score Rolling ({window}) aplicado a {col}.")
        return df

    def apply_time_encoding(self, df):
        """ Dirty 9 & 10: Codificación de Tiempo (Ciclos de Mercado) """
        # Asumimos que el índice es DatetimeIndex
        
        if self.cfg.FEATURES['USE_DAY_OF_WEEK']:
            # Seno/Coseno para mantener la ciclicidad (Lunes cerca de Domingo)
            day = df.index.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * day / 7)
            df['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        if self.cfg.FEATURES['USE_HOUR_OF_DAY']:
            hour = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
        print("   ✅ [TIME] Codificación Cíclica Temporal aplicada.")
        return df

    def apply_technical_indicators(self, df):
        """ Indicadores Clásicos (R3K Standard) usando Pandas-TA """
        
        # RSI
        if self.cfg.FEATURES['USE_RSI']:
            df['RSI'] = df.ta.rsi(length=self.cfg.FEATURES['RSI_PERIOD'])
            # Normalización del RSI (0-100 -> 0-1) para la Red Neuronal
            df['RSI_norm'] = df['RSI'] / 100.0

        # MACD
        if self.cfg.FEATURES['USE_MACD']:
            macd = df.ta.macd(
                fast=self.cfg.FEATURES['MACD_FAST'],
                slow=self.cfg.FEATURES['MACD_SLOW'],
                signal=self.cfg.FEATURES['MACD_SIGNAL']
            )
            df = pd.concat([df, macd], axis=1)

        # ADX (Filtro de Rango)
        if self.cfg.FEATURES['USE_ADX_FILTER']:
            adx = df.ta.adx(length=self.cfg.FEATURES['ADX_PERIOD'])
            df = pd.concat([df, adx], axis=1)
            # Normalizar ADX (aprox 0-60 -> 0-1)
            df['ADX_norm'] = df[f'ADX_{self.cfg.FEATURES["ADX_PERIOD"]}'] / 100.0

        # ATR (Para Position Sizing futuro)
        if self.cfg.FEATURES['USE_ATR']:
            df['ATR'] = df.ta.atr(length=self.cfg.FEATURES['ATR_PERIOD'])
            # ATR Relativo (ATR / Precio) para que sea comparable en el tiempo
            df['ATR_rel'] = df['ATR'] / df['Close']

        print("   ✅ [TECH] Indicadores Técnicos (TA-Lib) calculados.")
        return df

    def process_all(self, df):
        """ PIPELINE MAESTRO DE INGENIERÍA """
        print("💀 [FEATURE ENGINEER] Iniciando Refinería de Datos...")
        
        # 0. Copia de seguridad
        df = df.copy()
        
        # 1. Limpieza (Dirty 5)
        df = self.validate_data(df)
        
        # 2. Matemáticas Financieras (Dirty 6, 17)
        df = self.apply_log_returns(df)
        df = self.apply_parkinson_volatility(df)
        
        # 3. Normalización Local (Dirty 7)
        df = self.apply_rolling_zscore(df, col='Close', window=60)
        df = self.apply_rolling_zscore(df, col='Tick_Volume', window=60) # Volumen relativo
        
        # 4. Indicadores Técnicos
        df = self.apply_technical_indicators(df)
        
        # 5. Metadata Temporal (Dirty 9, 10)
        df = self.apply_time_encoding(df)
        
        # 6. Limpieza final de NaN generados por indicadores (Ventanas móviles)
        # Dirty 1: El Relleno Infinito (Preferimos dropna para training estricto, 
        # pero ffill para live. Aquí usamos dropna para asegurar calidad del dataset de entreno)
        initial_rows = len(df)
        df.dropna(inplace=True)
        lost_rows = initial_rows - len(df)
        
        print(f"💀 [PIPELINE COMPLETO] Filas procesadas: {len(df)}. (Nans eliminados: {lost_rows})")
        return df

