import zmq
import json
import time

# Simulation of MT5 sending a Tick
sample_tick = {
    "symbol": "XAUUSD",
    "ask": 2035.50,
    "bid": 2035.10,
    "balance": 10000.0,
    "equity": 10000.0,
    "has_position": False,
    "position_type": -1,
    "open_price": 0.0,
    "open_time": 0,
    "current_profit": 0.0
}

def test_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    print("Test Client: Sending Tick...")
    start_time = time.time()
    
    socket.send_string(json.dumps(sample_tick))
    
    response = socket.recv_string()
    end_time = time.time()
    
    print(f"Test Client: Received Response in {(end_time - start_time)*1000:.2f} ms")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_client()
