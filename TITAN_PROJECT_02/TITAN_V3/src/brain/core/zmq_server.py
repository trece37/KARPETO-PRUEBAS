import zmq
import json
import time
from datetime import datetime
from typing import Dict, Any

# Internal imports
# Internal imports
from ..api.brain_logic import predict, MarketData, TradeSignal

from .state_manager import StateManager
from .risk_engine import RiskEngine

class ZmqServer:
    def __init__(self, host="*", port=5555):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = False
        
        # [TAG: R3K_INTEGRATION]
        self.state = StateManager()
        self.risk = RiskEngine(self.state)
        print("✅ Risk Engine & State Manager Active")

    def start(self):
        """
        # [TAG: ANTIGRAVITY_ZMQ_SERVER]
        # Starts the ZeroMQ REP Server blocking loop.
        # Latency: ~3ms (vs 200ms+ HTTP).
        # Architecture: REP (Reply) matches with REQ (Request) in MT5.
        """
        bind_address = f"tcp://{self.host}:{self.port}"
        print(f"Antigravity ZMQ: Binding to {bind_address}")
        try:
            self.socket.bind(bind_address)
        except zmq.ZMQError as e:
            print(f"CRITICAL: Could not bind ZMQ socket: {e}")
            return

        self.running = True
        print("Antigravity ZMQ: ONLINE (Listening for Ticks...)")
        
        while self.running:
            try:
                # 1. Wait for Request (Tick)
                message = self.socket.recv_string()
                print(f"Received request: {message}")
                
                # 2. Process Request
                # [TAG: R3K_AUDIT]
                request_data = {}
                try:
                    request_data = json.loads(message)
                except json.JSONDecodeError:
                    request_data = {"raw": message}
                    
                self.state.log_event("ZMQ_REQUEST", request_data)

                # [TAG: RISK_CHECK]
                # Mocking account data for now (In prod, MT5 sends this)
                # If MT5 sends 'equity' and 'balance', use it.
                equity = request_data.get("equity", 10000.0) 
                balance = request_data.get("balance", 10000.0)
                
                allowed, reason = self.risk.validate_trade(equity, balance)
                
                response_dict = {}
                if not allowed:
                    response_dict = {
                        "action": "HOLD", 
                        "confidence": 0.0, 
                        "reason": f"VETO: {reason}"
                    }
                    print(f"⛔ RISK BLOCK: {reason}")
                else:
                    # Proceed to Brain Logic
                    response_dict = self.handle_message(message)
                
                # 3. Send Reply (Signal)
                self.socket.send_string(json.dumps(response_dict))
                self.state.log_event("ZMQ_RESPONSE", {"response": response_dict, "allowed": allowed})
                
            except KeyboardInterrupt:
                print("Antigravity ZMQ: Stopping...")
                self.running = False
                break
            except Exception as e:
                print(f"Antigravity ZMQ Error: {e}")
                # Always send a reply to prevent deadlock, even if error
                error_response = {
                    "action": "HOLD", 
                    "confidence": 0.0, 
                    "reason": f"Internal Error: {str(e)}"
                }
                self.socket.send_string(json.dumps(error_response))

    def handle_message(self, message: str) -> Dict[str, Any]:
        """
        # [TAG: DATA_BRIDGE]
        # Parsed JSON message from MT5 -> Pydantic Model -> Brain Logic.
        """
        try:
            data_dict = json.loads(message)
            
            # Map JSON to Pydantic Model
            market_data = MarketData(**data_dict)
            
            # Call Brain (Main Logic)
            signal: TradeSignal = predict(market_data)
            
            # Convert back to dict
            return signal.dict()
            
        except json.JSONDecodeError:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Invalid JSON"}
        except Exception as e:
            return {"action": "HOLD", "confidence": 0.0, "reason": f"Logic Error: {str(e)}"}

if __name__ == "__main__":
    server = ZmqServer()
    server.start()
