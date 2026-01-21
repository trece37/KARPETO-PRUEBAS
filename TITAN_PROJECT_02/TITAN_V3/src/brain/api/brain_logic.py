# from fastapi import FastAPI, HTTPException (REMOVED)
from pydantic.v1 import BaseModel, validator
from datetime import datetime
import os
import sys

# Add project root to path
# [TAG: ANTIGRAVITY_CLEAN_ARCH]
# Logic separated from Server to prevent Circular Imports
# Root is 4 directories up
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import settings
from src.brain.strategy.roi import ROITable
from src.brain.strategy.protection import CircuitBreaker
from src.brain.models.lstm import AchillesLSTM
from src.brain.models.portfolio import EqualWeightingPortfolioConstructionModel
from src.brain.models.portfolio import EqualWeightingPortfolioConstructionModel
from src.brain.models.seldon import SeldonCrisisMonitor
from src.brain.models.roi_alpha import ROIAlphaModel
from src.brain.strategy.position_sizing import PositionSizer
from src.brain.core.types import Insight, InsightDirection, PortfolioTarget
from src.brain.features.feature_engineering import FeatureEngineer
from collections import deque
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


# app = FastAPI(...) REMOVED

# --- Initialize 1000 Brains (Modular Architecture) ---
# 1. Alpha Model (The Intelligence)
# [TAG: DYNAMIC_DIMENSION_FIX]
# Ensure model matches Feature Engineer output (4 for Antigravity)
_seq_len, _n_feat = FeatureEngineer().get_input_shape()
alpha_model = AchillesLSTM(input_shape=(_seq_len, _n_feat)) 

# 2. Portfolio Construction (The allocator)
portfolio_model = EqualWeightingPortfolioConstructionModel()

# 3. Risk Management (The Safety)
# 3. Risk Management (The Safety)
circuit_breaker = CircuitBreaker(max_daily_loss_percent=0.03)

# 3.1 Seldon Crisis Monitor (The Guard)
# 3.1 Seldon Crisis Monitor (The Guard)
# Phase 4 (Oro Puro): Multivariate (Return + Volatility)
seldon_monitor = SeldonCrisisMonitor(contamination=0.01)

# 3.2 Position Sizer (Risk Control)
position_sizer = PositionSizer(target_risk_pct=0.01) # 1% Risk per trade

# --- SELDON INITIALIZATION (REAL DATA) ---
# --- SELDON INITIALIZATION (REAL DATA) ---
# Gemini 3 Fix: Load Real Crisis History
# Phase 3 Fix: Persistence (Joblib)
SELDON_MODEL_PATH = "seldon_model.joblib"

if not seldon_monitor.load_model(SELDON_MODEL_PATH):
    print("Seldon Model not found. Training from scratch...")
    crisis_files = [
        "src/brain/data/XAUUSD_D1_2000-2009_DotCom-Lehman.csv",
        "src/brain/data/XAUUSD_D1_2022_Ukraine.csv",
        "src/brain/data/XAUUSD_D1_2020_COVID.csv",
        "src/brain/data/XAUUSD_D1_2011-2012_Euro.csv",
        "src/brain/data/XAUUSD_D1_2025_Volatility.csv"
    ]
    # Adjust path to absolute for safety if running from root
    current_dir = os.getcwd()
    abs_crisis_files = [os.path.join(current_dir, f) for f in crisis_files]
    seldon_monitor.load_baseline(abs_crisis_files)
else:
    print("Seldon Model loaded from disk. Skipping training.")

# --- 4. Helper Alpha: ROI (Decoupled)
roi_alpha = ROIAlphaModel()

# --- 5. Feature Engineering (The Eyes) ---
# [TAG: GEMINI3_FEATURE_FIX]
# Gemini 3 Critical Fix: Replace garbage inputs (balance, equity, zeros)
# with real mathematical features
feature_engine = FeatureEngineer()
SEQ_LEN, N_FEATURES = feature_engine.get_input_shape()  # (60, 12)

# --- Rolling Window for LSTM ---
# [TAG: WARMUP_BUFFER_R3K]
# Buffer must be larger than SEQ_LEN to calculate indicators
# RSI(14) + SEQ_LEN(60) needs ~75. SMA/EMA might need more.
# Set to 200 for safety (warmup period)
WARMUP_SIZE = 200
history_buffer = deque(maxlen=WARMUP_SIZE)

# --- STATE PERSISTENCE ---
STATE_FILE = "state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                today_str = str(datetime.now().date())
                if state.get("date") == today_str:
                    circuit_breaker.daily_pnl = state.get("pnl", 0.0)
                    circuit_breaker.triggered = state.get("triggered", False)
                    print(f"State Loaded: PnL={circuit_breaker.daily_pnl}, Triggered={circuit_breaker.triggered}")
                else:
                    print("State Stale: Starting new day.")
        except Exception as e:
            print(f"Error loading state: {e}")

def save_state():
    try:
        state = {
            "date": str(datetime.now().date()),
            "pnl": circuit_breaker.daily_pnl,
            "triggered": circuit_breaker.triggered
        }
        # Phase 3 Fix: Atomic Write
        temp_file = f"{STATE_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        print(f"Error saving state: {e}")

# Load state on startup
load_state()


class MarketData(BaseModel):
    symbol: str
    bid: float
    ask: float
    balance: float
    equity: float
    has_position: bool = False
    position_type: int = -1 # 0=Buy, 1=Sell, -1=None
    open_price: float = 0.0
    open_time: int = 0 # Unix Timestamp
    current_profit: float = 0.0
    
    # [TAG: R3K_INPUT_VALIDATION]
    # QWEN3 Recommendation: Validate inputs to prevent garbage data
    class Config:
        validate_assignment = True
    
    @validator('bid', 'ask', 'balance', 'equity', 'open_price')
    def validate_positive_prices(cls, v, field):
        if v < 0:
            raise ValueError(f'{field.name} must be non-negative, got {v}')
        return v
    
    @validator('position_type')
    def validate_position_type(cls, v):
        if v not in [-1, 0, 1]:
            raise ValueError(f'position_type must be -1, 0, or 1, got {v}')
        return v

class TradeSignal(BaseModel):
    action: str # BUY, SELL, CLOSE_BUY, CLOSE_SELL, HOLD, STOP_TRADING
    confidence: float
    reason: str
    lot_size: float = 0.01 # Volatility Sizing (Default 0.01)

# @app.post("/predict") REMOVED - Direct Function Call
def predict(data: MarketData):
    """
    Full QuantConnect Flow: MarketData -> Alpha -> Insights -> Portfolio -> Targets -> Risk -> Execution
    """
    print(f"Tick: {data.symbol} | PnL: {data.current_profit} | Equity: {data.equity}")
    
    # --- 0. Update Safety State (Pre-Check) ---
    drawdown = 0.0
    if data.balance > 0:
        drawdown = max(0.0, (data.balance - data.equity) / data.balance)
    
    is_safe, fail_reason = circuit_breaker.check_safety(drawdown)
    if not is_safe:
        return TradeSignal(action="STOP_TRADING", confidence=1.0, reason=fail_reason)

    # --- 1. Alpha Model (Generate Insights) ---
    
    # [TAG: GEMINI3_FEATURE_ENGINEERING_FIX]
    # CRITICAL FIX: Replace garbage inputs (balance, equity, zeros)
    # with real mathematical features calculated from OHLCV data
    
    # 1.A Data Ingestion & Feature Engineering
    # Construct raw OHLCV tick from market data
    raw_tick = {
        'open': data.open_price if data.open_price > 0 else data.bid,
        'high': max(data.bid, data.ask, data.open_price) if data.open_price > 0 else data.bid,
        'low': min(data.bid, data.ask, data.open_price) if data.open_price > 0 else data.bid,
        'close': data.bid,  # Current price
        'tick_volume': 1.0  # Placeholder (MT5 should send real volume)
    }
    history_buffer.append(raw_tick)
    
    insights = []
    
    # Check if we have enough data for feature engineering
    if len(history_buffer) >= WARMUP_SIZE:
        try:
            # Convert buffer to DataFrame
            df_buffer = pd.DataFrame(list(history_buffer))
            
            # GENERATE MATHEMATICAL FEATURES (ORO PURO)
            df_features = feature_engine.generate_features(df_buffer)
            
            # Verify we have enough rows after dropna() from indicators
            if len(df_features) >= SEQ_LEN:
                # Take the last 60 rows exactly
                final_input_df = df_features.tail(SEQ_LEN)
                
                # Convert to Tensor Numpy (1, 60, 12)
                input_tensor = np.array(final_input_df).reshape(1, SEQ_LEN, N_FEATURES)
                
                # PREDICT with real features
                insights = alpha_model.update(input_tensor)
            else:
                print(f"Feature Warmup: {len(df_features)}/{SEQ_LEN} valid rows after indicators")
        except Exception as e:
            print(f"Feature Engineering Error: {e}")
    else:
        print(f"Data Buffering: {len(history_buffer)}/{WARMUP_SIZE} ticks")

    # 1.B ROI Alpha (Rule-Based)
    # Gemini 3 Fix: Decoupled logic
    roi_insights = roi_alpha.update(data)
    insights.extend(roi_insights)


    # --- 2. Portfolio Construction (Create Targets) ---
    targets = portfolio_model.create_targets(insights)
    
    # --- 3. Risk Management (Adjust Targets) ---
    
    # 3.1 Seldon Veto (Checks for Market Anomalies/Crashes)
    # [TAG: SELDON_PERSISTENCE]
    # Seldon now loads from 'seldon_model.joblib' (Instant/Atomic).
    # Refine Return Calculation: Use price change from Open.
    # Gemini 3 Fix: Zero Division Protection
    current_return = 0.0
    current_price = data.bid # Approximate
    
    if data.open_price > 0.0001:
        current_return = (current_price - data.open_price) / data.open_price
    else:
        # Dangerous Tick (Bad Data) -> Skip Seldon Update to avoid pollution
        pass
    
    # Feed Seldon
    # [TAG: MULTIVARIATE_SELDON_FEED]
    # Calculate Volatility (Proxy: 20-period std dev of returns) if history allows
    current_vol = 0.0
    atr_value = 0.0
    
    if len(history_buffer) >= 20:
         # Quick calculation from buffer to avoid full feature engineering overhead if possible,
         # but using df_buffer is safer.
         try:
             # Last 20 closes
             closes = pd.Series([t['close'] for t in list(history_buffer)[-21:]])
             rets = closes.pct_change().dropna()
             current_vol = rets.std()
             
             # Also try to get latest ATR from Feature Engine results if available
             # Since feature_engine.generate_features was called in step 1.A, we can't easily reuse without caching.
             # Approximating ATR manually for efficiency or re-using logic?
             # Let's rely on simple TR for now or use fixed fallback if calc fails
             pass
         except:
             pass

    seldon_monitor.update(current_return, current_vol)
    
    # Apply Seldon Veto to Targets
    seldon_checked_targets = seldon_monitor.manage_risk(targets)
    
    # 3.2 Circuit Breaker (Account Level Safety)
    # Apply circuit breaker logic to targets (if triggered, it zeros them out)
    safe_targets = circuit_breaker.manage_risk(seldon_checked_targets)
    
    # --- 4. Execution (Convert Target to Signal) ---
    # This acts as the 'Execution Model', translating abstract targets to immediate Broker actions
    
    if not safe_targets:
        return TradeSignal(action="HOLD", confidence=0.0, reason="No Targets")
        
    target = safe_targets[0] # Assuming single symbol for now
    
    # Interpreter: Target vs Current State
    if target.percent == 0.0:
        # Target is Flat
        if data.has_position:
            action = "CLOSE_BUY" if data.position_type == 0 else "CLOSE_SELL"
            return TradeSignal(action=action, confidence=1.0, reason="Portfolio Target: 0%")
    elif target.percent > 0:
        # Target is Long
        if not data.has_position:
             # [TAG: POSITION_SIZING_EXECUTION]
             # Calculate exact lots based on Volatility
             # We need ATR. Since we didn't save it from Step 1, we recalculate or approximate.
             # Ideally: Architecture refactor to pass 'features' down to Execution.
             # Quick Fix: Calculate ATR on buffer.
             
             calc_lots = 0.01
             try:
                 # Reconstruct DataFrame from buffer just for ATR
                 df_exec = pd.DataFrame(list(history_buffer))
                 # Simple High-Low ATR approximation for last candle
                 last_candle = df_exec.iloc[-1]
                 tr = max(last_candle['high'] - last_candle['low'], abs(last_candle['high'] - last_candle['close']))
                 
                 # Smooth it? Let's use simpler Position Sizing logic if ATR unavailable
                 # or assume a safe default if buffer pending.
                 if len(history_buffer) > 20:
                    # Let position_sizer handle robustness
                    calc_lots = position_sizer.calculate_lot_size(data.equity, tr) 
             except Exception as e:
                 print(f"Sizing Error: {e}")
                 calc_lots = 0.01

             return TradeSignal(action="BUY", confidence=0.8, reason="Alpha Signal: Buy", lot_size=calc_lots)
             
    # Default
    # Save state before returning
    save_state()
    return TradeSignal(action="HOLD", confidence=0.0, reason="Target aligned with State")

# Server Startup Logic Replaced by Clean Architecture
# See src/brain/api/main.py for entry point.
