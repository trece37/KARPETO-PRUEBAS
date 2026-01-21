"""
ADF Test (Augmented Dickey-Fuller) - Data Stationarity Validator
Phase 4.2: Oro Puro R3K Compliance

This module validates that financial time series data is stationary before LSTM training.
Non-stationary data (trending prices) leads to overfitting. ADF test detects this.

References: Oro Puro (00000.Todas las fuentes - oropuro1.MD)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict

class StationarityValidator:
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize ADF Stationarity Validator.
        
        Args:
            significance_level: P-value threshold (default 0.05 = 95% confidence)
        """
        # [TAG: R3K_STATIONARITY_THRESHOLD]
        self.significance_level = significance_level
        
    def test_stationarity(self, data: pd.Series, label: str = "Series") -> Dict:
        """
        # [TAG: ADF_TEST_R3K]
        Perform Augmented Dickey-Fuller test to check stationarity.
        
        Null Hypothesis (H0): Series has a unit root (non-stationary)
        Alternative (H1): Series is stationary
        
        Args:
            data: Time series data (e.g., prices or returns)
            label: Name of the series for logging
            
        Returns:
            Dict with test results and recommendation
        """
        # Run ADF test
        result = adfuller(data.dropna(), autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # [TAG: ADF_INTERPRETATION]
        # If p-value < 0.05, reject H0 → stationary (GOOD)
        # If p-value > 0.05, fail to reject H0 → non-stationary (BAD)
        is_stationary = p_value < self.significance_level
        
        verdict = {
            "series": label,
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "critical_values": critical_values,
            "is_stationary": is_stationary,
            "recommendation": self._get_recommendation(is_stationary)
        }
        
        return verdict
    
    def _get_recommendation(self, is_stationary: bool) -> str:
        """Generate action recommendation based on stationarity."""
        if is_stationary:
            return "✅ PASS: Data is stationary. Safe for LSTM training."
        else:
            return "❌ FAIL: Data is non-stationary. Apply differencing (use returns instead of prices)."
    
    def validate_and_transform(self, prices: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        # [TAG: R3K_AUTO_TRANSFORM]
        Test stationarity and auto-transform if needed.
        
        Workflow:
        1. Test raw prices
        2. If non-stationary, convert to log returns
        3. Test returns
        4. Return the stationary series
        
        Args:
            prices: Raw price series
            
        Returns:
            (stationary_series, test_results)
        """
        print("=" * 60)
        print("ADF STATIONARITY VALIDATION (R3K)")
        print("=" * 60)
        
        # Test 1: Raw Prices
        print("\n[1/2] Testing raw prices...")
        price_test = self.test_stationarity(prices, label="Prices")
        
        if price_test["is_stationary"]:
            print(f"P-value: {price_test['p_value']:.6f} → {price_test['recommendation']}")
            return prices, price_test
        
        # Test 2: Log Returns (Differencing)
        print(f"P-value: {price_test['p_value']:.6f} → Non-stationary detected")
        print("\n[2/2] Applying differencing (log returns)...")
        
        returns = np.log(prices / prices.shift(1)).dropna()
        return_test = self.test_stationarity(returns, label="Log Returns")
        
        print(f"P-value: {return_test['p_value']:.6f} → {return_test['recommendation']}")
        
        if not return_test["is_stationary"]:
            print("⚠️ WARNING: Even returns are non-stationary. Data may be toxic.")
        
        return returns, return_test

if __name__ == "__main__":
    # Example Test with Synthetic Data
    print("Testing ADF Validator with synthetic data...\n")
    
    # Generate non-stationary data (random walk with drift)
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(500) * 2 + 0.1))
    
    validator = StationarityValidator()
    stationary_data, results = validator.validate_and_transform(prices)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Output series: {results['series']}")
    print(f"Stationary: {results['is_stationary']}")
