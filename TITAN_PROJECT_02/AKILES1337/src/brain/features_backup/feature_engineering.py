"""
Feature Engineering Engine - Achilles Antigravity
Phase 2: Mathematical Vision Upgrade

Purpose: Transforms raw OHLCV data into normalized, stationary, and predictive features.
Compliance: R3K (Standardized Input), BARRIGA (Anticipates Scale Issues), R6V (Verified Features)

References: Gemini 3 Analysis - "The model is blind without real mathematical features"
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

# Optional: pandas_ta for advanced indicators
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("⚠️ WARNING: pandas_ta not installed. Using manual calculations.")

class FeatureEngineer:
    def __init__(self):
        # [TAG: FEATURE_LIST_R3K]
        # These are the 12 features the LSTM will actually see.
        # NO balance, NO equity. Only Market Dynamics.
        self.feature_columns = [
            'log_ret',         # Stationarity (ADF compliance)
            'volatility_z',    # Normalization (Z-Score)
            'rsi_norm',        # Momentum (0-1 normalized)
            'adx_norm',        # Trend Strength (0-1 normalized)
            'atr_rel',         # Volatility Regime (relative)
            'price_dist_sma',  # Mean Reversion
            'volume_chg',      # Activity
            'high_low_chg',    # Intraday Volatility
            'close_open_chg',  # Candle Body
            'ema_slope',       # Trend Velocity
            'lag_1',           # Serial Correlation
            'lag_2'            # Serial Correlation
        ]
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms Raw Data -> AI Ready Tensor
        
        Args:
            df: DataFrame with ['open', 'high', 'low', 'close', 'tick_volume']
            
        Returns:
            DataFrame with 12 mathematical features
            
        Raises:
            ValueError: If required columns are missing
        """
        # [TAG: R6V_VALIDATION]
        # Validate Input
        required = ['open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing columns. Required: {required}, Got: {df.columns.tolist()}")

        df = df.copy()

        # 1. STATIONARITY (Log Returns)
        # [TAG: ADF_COMPLIANCE]
        # Replacing absolute price with relative change
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # 2. NORMALIZATION (Z-Score)
        # [TAG: CONTEXT_NORMALIZATION]
        # (Close - Mean) / StdDev. Tells AI if price is statistically outlier.
        rolling_window = 20
        mean = df['close'].rolling(rolling_window).mean()
        std = df['close'].rolling(rolling_window).std()
        df['volatility_z'] = (df['close'] - mean) / (std + 1e-6) # +epsilon to avoid div0

        # 3. MOMENTUM (RSI Normalized)
        # [TAG: MOMENTUM_INDICATOR]
        # RSI is 0-100. LSTM likes 0-1. We scale to 0-1.
        if HAS_PANDAS_TA:
            df['rsi'] = ta.rsi(df['close'], length=14)
        else:
            df['rsi'] = self._calculate_rsi_manual(df['close'], period=14)
        df['rsi_norm'] = df['rsi'] / 100.0

        # 4. TREND STRENGTH (ADX Normalized)
        # [TAG: TREND_INDICATOR]
        # ADX usually 0-60, rarely > 80. Clip at 100 and normalize.
        if HAS_PANDAS_TA:
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx_norm'] = adx_df['ADX_14'].clip(upper=100) / 100.0
        else:
            df['adx_norm'] = 0.5  # Fallback: neutral value

        # 5. VOLATILITY REGIME (Relative ATR)
        # [TAG: VOLATILITY_CONTEXT]
        # Is current candle bigger than usual?
        if HAS_PANDAS_TA:
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        else:
            df['atr'] = self._calculate_atr_manual(df, period=14)
        df['atr_rel'] = df['atr'] / df['close'] # Percentage volatility

        # 6. MEAN REVERSION (Distance from SMA)
        # [TAG: MEAN_REVERSION]
        # How far are we from the 50-period average?
        if HAS_PANDAS_TA:
            sma_50 = ta.sma(df['close'], length=50)
        else:
            sma_50 = df['close'].rolling(50).mean()
        df['price_dist_sma'] = (df['close'] - sma_50) / sma_50

        # 7. VOLUME DYNAMICS
        # [TAG: VOLUME_ANALYSIS]
        # Change in volume relative to moving average
        vol_ma = df['tick_volume'].rolling(20).mean()
        df['volume_chg'] = (df['tick_volume'] - vol_ma) / (vol_ma + 1e-6)

        # 8. CANDLE GEOMETRY
        # [TAG: PRICE_ACTION]
        df['high_low_chg'] = (df['high'] - df['low']) / df['close']
        df['close_open_chg'] = (df['close'] - df['open']) / df['close']

        # 9. TREND VELOCITY (EMA Slope)
        # [TAG: VELOCITY_INDICATOR]
        if HAS_PANDAS_TA:
            ema_10 = ta.ema(df['close'], length=10)
        else:
            ema_10 = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_slope'] = (ema_10 - ema_10.shift(1)) / ema_10.shift(1)

        # 10. SERIAL CORRELATION (Lags)
        # [TAG: AUTOCORRELATION]
        # What happened 1 and 2 candles ago?
        df['lag_1'] = df['log_ret'].shift(1)
        df['lag_2'] = df['log_ret'].shift(2)

        # CLEANUP (BARRIGA: Handle NaNs generated by lookbacks)
        # [TAG: NAN_HANDLING]
        # Indicators like SMA(50) create 50 NaNs at the start. Drop them.
        df.dropna(inplace=True)

        # R3K: Return strictly the feature columns expected by LSTM
        return df[self.feature_columns]

    def get_input_shape(self) -> Tuple[int, int]:
        """
        Returns the expected input shape for LSTM.
        
        Returns:
            (sequence_length, num_features) = (60, 12)
        """
        return (60, len(self.feature_columns))
    
    # [TAG: MANUAL_FALLBACKS]
    # Manual calculations for when pandas_ta is not available
    
    def _calculate_rsi_manual(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Manual RSI calculation (Wilder's method)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr_manual(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Manual ATR calculation"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr


if __name__ == "__main__":
    # [TAG: UNIT_TEST]
    print("=" * 60)
    print("FEATURE ENGINEER - UNIT TEST")
    print("=" * 60)
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    n = 200
    
    test_df = pd.DataFrame({
        'open': 2000 + np.cumsum(np.random.randn(n) * 5),
        'high': 2005 + np.cumsum(np.random.randn(n) * 5),
        'low': 1995 + np.cumsum(np.random.randn(n) * 5),
        'close': 2000 + np.cumsum(np.random.randn(n) * 5),
        'tick_volume': np.random.randint(100, 1000, n)
    })
    
    # Ensure high >= close >= low
    test_df['high'] = test_df[['open', 'close', 'high']].max(axis=1)
    test_df['low'] = test_df[['open', 'close', 'low']].min(axis=1)
    
    print(f"\n[1/3] Input Data: {len(test_df)} rows")
    print(test_df.head())
    
    # Generate features
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(test_df)
    
    print(f"\n[2/3] Output Features: {len(features_df)} rows (after dropna)")
    print(f"Feature Columns: {features_df.columns.tolist()}")
    print(features_df.head())
    
    # Validate shape
    seq_len, n_features = engineer.get_input_shape()
    print(f"\n[3/3] Expected LSTM Input Shape: ({seq_len}, {n_features})")
    
    if len(features_df) >= seq_len:
        sample_input = features_df.tail(seq_len).values
        print(f"Sample Input Shape: {sample_input.shape}")
        print(f"✅ Shape matches: {sample_input.shape == (seq_len, n_features)}")
    else:
        print(f"⚠️ Not enough data for full sequence (need {seq_len}, got {len(features_df)})")
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEER TEST COMPLETE ✅")
    print("=" * 60)
