"""
ZMQ High Frequency Bridge - Achilles Antigravity
Architecture: Asynchronous Threaded Server
"""
import zmq
import json
import threading
import time

class ZMQBridge:
    def __init__(self, pull_port=5555, pub_port=5556):
        self.context = zmq.Context()
        # SOCKET PULL (Recibe datos sin bloquear)
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{pull_port}")
        
        # SOCKET PUB (Envía órdenes)
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{pub_port}")
        
        self.running = False
        self.last_tick_time = time.time()
        print(f"🚀 ZMQ BRIDGE: Listening PULL:{pull_port} | PUB:{pub_port}")

    def start_listening(self, callback_function):
        """Inicia el hilo demonio de escucha."""
        self.running = True
        thread = threading.Thread(target=self._loop, args=(callback_function,))
        thread.daemon = True
        thread.start()
        print("✅ ZMQ Thread Started.")

    def _loop(self, callback):
        while self.running:
            try:
                # recv_json es bloqueante, pero está en su propio hilo.
                msg = self.pull_socket.recv_json()
                self.last_tick_time = time.time()
                
                # Callback al cerebro principal
                decision = callback(msg)
                
                if decision:
                    self.send_signal(decision)
                    
            except Exception as e:
                print(f"🔥 ZMQ LOOP ERROR: {e}")
                time.sleep(0.001)

    def send_signal(self, signal_dict):
        """Envía el payload JSON al worker."""
        message = {'topic': 'TRADE', 'data': signal_dict}
        self.pub_socket.send_json(message)
        print(f"📡 ZMQ SENT: {signal_dict.get('action')}")

    def check_health(self):
        if time.time() - self.last_tick_time > 15:
            print("⚠️ ZMQ WARNING: No ticks in 15s. Check MT5.")
