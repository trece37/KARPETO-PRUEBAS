"""
WFO Test Script
Phase 4: Testing Walk Forward Optimization with Synthetic Data

This script verifies that the WFO engine works correctly before applying it to real LSTM.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain.validation.wfo_validator import WFOValidator

def generate_synthetic_price_data(days: int = 1000, start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Generate synthetic price data for testing WFO.
    
    Args:
        days: Number of days to generate
        start_date: Starting date
        
    Returns:
        DataFrame with OHLC data
    """
    print(f"Generating {days} days of synthetic data...")
    
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Generate random walk price
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # Mean return 0.05%, Std 2%
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.99, 1.01, days),
        'high': prices * np.random.uniform(1.00, 1.02, days),
        'low': prices * np.random.uniform(0.98, 1.00, days),
        'close': prices,
        'volume': np.random.randint(1000, 10000, days)
    }, index=dates)
    
    return df

def test_wfo_engine():
    """
    Test the WFO engine with synthetic data.
    """
    print("=" * 60)
    print("PHASE 4: WFO VALIDATOR TEST")
    print("=" * 60)
    
    # 1. Generate Test Data
    data = generate_synthetic_price_data(days=800, start_date="2020-01-01")
    print(f"Dataset: {data.index.min()} to {data.index.max()}")
    print(f"Total rows: {len(data)}")
    
    # 2. Initialize WFO Validator
    validator = WFOValidator(config_path="wfo_config.yaml")
    
    # 3. Generate Windows
    windows = validator.generate_windows(data)
    
    if not windows:
        print("ERROR: No windows generated!")
        return False
    
    print(f"\nSUCCESS: Generated {len(windows)} WFO windows")
    
    # 4. Inspect First Window
    print("\n--- First Window Inspection ---")
    in_sample, out_of_sample = windows[0]
    print(f"In-Sample: {len(in_sample)} rows ({in_sample.index.min()} to {in_sample.index.max()})")
    print(f"Out-of-Sample: {len(out_of_sample)} rows ({out_of_sample.index.min()} to {out_of_sample.index.max()})")
    
    # 5. Verify No Data Leakage
    if in_sample.index.max() >= out_of_sample.index.min():
        print("ERROR: Data leakage detected! In-Sample and Out-of-Sample overlap!")
        return False
    
    print("✅ No data leakage. In-Sample ends before Out-of-Sample starts.")
    
    print("\n" + "=" * 60)
    print("WFO ENGINE: VALIDATED ✅")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_wfo_engine()
    sys.exit(0 if success else 1)
