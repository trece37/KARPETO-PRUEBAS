import asyncio
import zmq.asyncio
import json
import random
import time
import os
import sys
import pandas as pd
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Configuración de Rutas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Carga de Módulos
try:
    from src.brain.features.feature_engineering import FeatureEngineer
    from src.brain.models.lstm import AchillesLSTM
    from src.brain.models.seldon import SeldonCrisisMonitor
    from src.brain.strategy.protection import CircuitBreaker
    from src.brain.core.types import InsightDirection
except ImportError as e:
    print(f"❌ ERROR DE IMPORTACIÓN: {e}")
    sys.exit(1)

class SentinelServer:
    def __init__(self, sub_port=5556, push_port=5557):
        self.sub_port = sub_port
        self.push_port = push_port
        self.context = zmq.asyncio.Context()
        
        self.sub_socket = self.context.socket(zmq.SUB)
        self.push_socket = self.context.socket(zmq.PUSH)
        
        self.warmup_size = 200
        self.history = deque(maxlen=self.warmup_size)
        
        # Componentes Pesados (Math Heavy)
        self.engineer = FeatureEngineer()
        self.brain = AchillesLSTM()
        self.seldon = SeldonCrisisMonitor()
        self.circuit_breaker = CircuitBreaker()
        
        # Executor para no bloquear el Event Loop
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def start(self):
        print(f"🦅 SENTINEL SERVER (ASYNC PRO): INICIANDO (SUB:{self.sub_port} | PUSH:{self.push_port})")
        
        # Bind (Servidor) vs Connect?
        # En topología PUB/SUB estándar:
        # MT5 (PUB) -> BIND 5556
        # Python (SUB) -> CONNECT 5556
        # MT5 (PULL) -> BIND 5557
        # Python (PUSH) -> CONNECT 5557
        # Confirmamos que Sentinel actúa como cliente de la conexión TCP (Connect)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.sub_port}")
        self.sub_socket.subscribe("")
        self.push_socket.connect(f"tcp://127.0.0.1:{self.push_port}")
        
        print("✅ SISTEMA NERVIOSO CONECTADO. ESCUCHANDO MERCADO...")
        
        try:
            await self.listen_loop()
        except Exception as e:
            print(f"🔥 FALLO CRÍTICO EN SENTINEL: {e}")
        finally:
            self.sub_socket.close()
            self.push_socket.close()
            self.context.term()
            self.executor.shutdown()

    async def listen_loop(self):
        while True:
            # Recepción Asíncrona (Non-Blocking)
            message = await self.sub_socket.recv_string()
            data = json.loads(message)
            
            # Disparamos la tarea sin esperar (Fire & Forget logic for receiving, but await for processing path)
            # Queremos que el loop siga libre para recibir el siguiente tick inmediatamente
            asyncio.create_task(self.process_tick_pipeline(data))

    async def process_tick_pipeline(self, data):
        # 1. Update State (Fast)
        self.history.append({
            'open': data['bid'],
            'high': data['bid'],
            'low': data['bid'],
            'close': data['bid'],
            'volume': 1.0
        })
        
        # 2. Heavy Lifting in Thread (Slow Math)
        if len(self.history) >= self.warmup_size:
            loop = asyncio.get_running_loop()
            # Pasamos una copia de la historia para evitar condiciones de carrera si append ocurre durante el cálculo
            history_snapshot = list(self.history)
            
            # Ejecutar lógica pesada fuera del loop principal
            decision = await loop.run_in_executor(
                self.executor, 
                self.cpu_bound_logic, 
                history_snapshot, 
                data.get('drawdown', 0.0)
            )
            
            if decision:
                await self.send_order(decision['action'], decision['reason'])

    def cpu_bound_logic(self, history_data, drawdown):
        """Bloque de cálculo pesado. Se ejecuta en otro hilo."""
        try:
            # A. Protection
            if not self.circuit_breaker.check_safety(drawdown):
                return {"action": "STOP_ALL", "reason": "Circuit Breaker"}

            # B. Features
            df = pd.DataFrame(history_data)
            features = self.engineer.generate_features(df)
            
            # C. Seldon Check
            if self.seldon.check(features.iloc[-1]):
                return None # Veto

            # D. Inference
            input_seq = features.tail(60) # Asumiendo lookback 60
            prediction = self.brain.predict(input_seq)
            
            # E. VOM Logic
            if prediction['confidence'] > 0.80:
                 if prediction['direction'] == "UP": return {"action": "BUY", "reason": "AI_UP"}
                 if prediction['direction'] == "DOWN": return {"action": "SELL", "reason": "AI_DOWN"}
            
            return None
        except Exception as e:
            # print(f"Math Error: {e}") # Debug only
            return None

    async def send_order(self, action, reason):
        # Humanize Latency (R3K)
        await asyncio.sleep(random.uniform(0.02, 0.05))
        
        order = {
            "symbol": "XAUUSD",
            "action": action,
            "reason": reason,
            "timestamp": time.time()
        }
        await self.push_socket.send_string(json.dumps(order))
        print(f"🚀 ORDEN: {action} ({reason})")

if __name__ == "__main__":
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        server = SentinelServer()
        asyncio.run(server.start())
    except KeyboardInterrupt:
        pass
