from fastapi.testclient import TestClient
from src.brain.api.main import app
import pytest

client = TestClient(app)

def test_seldon_normal_market():
    # 1. Simulate Normal Market (Near 0 return)
    payload = {
        "symbol": "XAUUSD",
        "bid": 2000.0,
        "ask": 2000.5,
        "balance": 10000.0,
        "equity": 10000.0,
        "has_position": True,
        "position_type": 0, # Buy
        "open_price": 2000.0, # 0% return
        "open_time": 1700000000,
        "current_profit": 0.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("\n[Normal] Response:", data)
    # Should probably be HOLD or SELL based on ROI, but definitely not forced liquidate by Seldon (unless ROI triggers)
    # The key is that it didn't crash.

def test_seldon_crash_market():
    # 2. Simulate CRASH (Massive drop vs Open)
    # Open: 2000, Current: 1800 (-10%)
    # This should be > 3 sigma vs the N(0, 0.1%) dummy distribution
    payload = {
        "symbol": "XAUUSD",
        "bid": 1800.0,
        "ask": 1800.5,
        "balance": 9000.0, # Equity drop reflected
        "equity": 9000.0,
        "has_position": True,
        "position_type": 0, # Buy
        "open_price": 2000.0, 
        "open_time": 1700000000,
        "current_profit": -1000.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("\n[Crash] Response:", data)
    
    # Expectation: Seldon detects anomaly -> Returns Target 0% -> Execution converts to CLOSE_BUY
    assert data["action"] == "CLOSE_BUY"
    # Note: Reason might be "Portfolio Target: 0%" which is what Seldon forces.

if __name__ == "__main__":
    test_seldon_normal_market()
    test_seldon_crash_market()
