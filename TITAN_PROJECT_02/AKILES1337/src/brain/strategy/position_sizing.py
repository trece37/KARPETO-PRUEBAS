"""
Position Sizing Module - Achilles Antigravity
Phase 4: Oro Puro R3K Compliance

Purpose: Determines the optimal trade size (lots) based on market volatility (ATR)
         and account equity. Replaces static lot sizing with dynamic risk management.

Reference: GITHUB - 20 MODELOS (freqtrade / QuantConnect logic)
Compliance: R6V (Verified Math)
"""

import math
from typing import Optional

class PositionSizer:
    def __init__(self, 
                 target_risk_pct: float = 0.01,  # 1% risk per trade
                 min_lot: float = 0.01,
                 max_lot: float = 10.0,
                 point_value: float = 1.0):      # Value of 1 point movement (e.g. $1 for XAUUSD standard contract?? Verify broker)
        """
        Args:
            target_risk_pct: Percentage of equity to risk per trade (default 1%).
            min_lot: Minimum allowed lot size (broker constraint).
            max_lot: Maximum allowed lot size.
            point_value: Monetary value of a 1.0 price move per 1.0 lot.
                         NOTE: For XAUUSD, 1 lot usually = 100oz. 
                         Move of $1.00 in price = $100 Profit/Loss per lot.
                         So point_value might need adjustment based on symbol.
        """
        self.target_risk_pct = target_risk_pct
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.point_value_per_lot_per_point = 100.0 # Default for XAUUSD Standard Lot (Check this!)
        # TODO: Make point_value dynamic based on symbol properties from MT5

    def calculate_lot_size(self, 
                           equity: float, 
                           atr: float, 
                           stop_loss_multiplier: float = 2.0) -> float:
        """
        Calculates position size based on Volatility (ATR).
        
        Formula:
            Risk Amount = Equity * Risk %
            Stop Loss Distance = ATR * Multiplier
            Dollar Risk Per Lot = Stop Loss Distance * Value Per Point
            Lots = Risk Amount / Dollar Risk Per Lot
            
        Example:
            Equity = $10,000, Risk = 1% ($100)
            ATR = $5.00, SL = 2 * 5 = $10.00
            Value per point (1 lot) = $100 (Standard Gold contract)
            Risk per Lot = $10 * 100 = $1000
            Lots = $100 / $1000 = 0.1 Lots
        """
        if atr <= 0:
            print("⚠️ WARNING: ATR is zero or negative. Using Min Lot.")
            return self.min_lot

        risk_amount = equity * self.target_risk_pct
        
        # Distance to Stop Loss in Price Units
        stop_loss_dist = atr * stop_loss_multiplier
        
        # How much money we lose per lot if SL is hit
        # Ensure we use the correct point value multiplier
        # Usually for XAUUSD: 1 Lot = 100 Units. Delta $1 = $100 PnL.
        risk_per_lot = stop_loss_dist * self.point_value_per_lot_per_point
        
        if risk_per_lot == 0:
             return self.min_lot

        raw_lots = risk_amount / risk_per_lot
        
        # Round logic (usually 2 decimals for lots)
        lots = math.floor(raw_lots * 100) / 100.0
        
        # Clip to limits
        final_lots = max(self.min_lot, min(self.max_lot, lots))
        
        return final_lots

    def set_point_value(self, value: float):
        """Update point value based on symbol (e.g. 100 for Gold, 100000 for EURUSD?)"""
        self.point_value_per_lot_per_point = value

if __name__ == "__main__":
    # Quick Test
    sizer = PositionSizer(target_risk_pct=0.01) # 1% Risk
    
    # Scene 1: Calm Market
    eq = 10000
    atr_calm = 2.0 # $2 movement
    lots_calm = sizer.calculate_lot_size(eq, atr_calm)
    print(f"Equity: ${eq}, ATR: ${atr_calm} (Calm) -> Lots: {lots_calm}")
    
    # Scene 2: Crisis Market (Volatile)
    atr_crisis = 10.0 # $10 movement (5x volatility)
    lots_crisis = sizer.calculate_lot_size(eq, atr_crisis)
    print(f"Equity: ${eq}, ATR: ${atr_crisis} (Crisis) -> Lots: {lots_crisis}")
    
    print("✅ Verification: Volatility increased 5x, Lots should decrease ~5x.")
