from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.brain.api.main import app

client = TestClient(app)

def test_roi_trigger():
    # Simulate a trade open for 15 mins with 4% profit
    # Config says: >10 min needs >3% profit using Freqtrade logic in roi.py
    
    payload = {
        "symbol": "XAUUSD",
        "bid": 2000.0,
        "ask": 2001.0,
        "balance": 10000.0,
        "equity": 10400.0,
        "has_position": True,
        "position_type": 0, # BUY
        "open_price": 1923.0, # (2000 - 1923)/1923 = 4% profit
        "open_time": 1702390000, # Old timestamp
        "current_profit": 400.0
    }
    
    # We need to hack the open_time to be relative to NOW for the test to work deterministically
    import time
    now_ts = int(time.time())
    payload["open_time"] = now_ts - (15 * 60) # 15 mins ago
    
    response = client.post("/predict", json=payload)
    data = response.json()
    
    print(f"ROI Test: {data}")
    assert data["action"] == "CLOSE_BUY"
    assert "ROI Target Reached" in data["reason"]

def test_circuit_breaker():
    # Simulate heavy drawdown (Equity < Balance - 5%)
    # Limit is 3%
    
    payload = {
        "symbol": "XAUUSD",
        "bid": 2000.0,
        "ask": 2001.0,
        "balance": 10000.0,
        "equity": 9500.0, # 5% Drawdown
        "has_position": False
    }
    
    response = client.post("/predict", json=payload)
    data = response.json()
    
    print(f"Breaker Test: {data}")
    assert data["action"] == "STOP_TRADING"
    assert "Circuit Breaker TRIGGERED" in data["reason"]

if __name__ == "__main__":
    print("Running Tests...")
    test_roi_trigger()
    test_circuit_breaker()
    print("ALL TESTS PASSED")
