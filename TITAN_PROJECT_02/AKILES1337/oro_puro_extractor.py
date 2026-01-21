import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import time

def extract_oro_puro_dataset():
    print("🦅 [ANTIGRAVITY] BUSCANDO EL LÍMITE DE TUS DATOS...")
    if not mt5.initialize():
        print(f"❌ Error al inicializar MT5: {mt5.last_error()}")
        return

    info = mt5.terminal_info()
    if info:
        print(f"📊 Terminal: {info.name} | Max Bars Config: {info.maxbars}")

    symbols = mt5.symbols_get()
    symbol = next((s.name for s in symbols if "XAUUSD" in s.name.upper() or "GOLD" in s.name.upper()), None)
    
    if not symbol:
        print("❌ Símbolo no encontrado.")
        mt5.shutdown()
        return

    print(f"✅ Símbolo: {symbol}")
    mt5.symbol_select(symbol, True)
    
    # Bucle de búsqueda del tamaño máximo
    test_sizes = [1000000, 500000, 100000, 50000, 25000]
    rates = None
    
    for size in test_sizes:
        print(f"� Probando extracción de {size:,} barras M1...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, size)
        if rates is not None and len(rates) > 0:
            print(f"🎯 ¡CONSEGUIDO! Se han extraído {len(rates):,} barras.")
            break

    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        output_dir = r"c:\Users\David\AchillesTraining\00_FACTORY\TITAN_V3\data"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"XAUUSD_M1_MASTER_REAL_DATA.csv"
        filepath = os.path.join(output_dir, filename)
        
        df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_csv(filepath, index=False)
        print(f"💎 DATASET GUARDADO: {len(df):,} barras en {filename}")
        print(f"� Cobertura: {df['time'].min()} hasta {df['time'].max()}")
        
        # También intentamos M5 para más profundidad
        print(f"🚀 Intentando obtener M5 para mayor profundidad temporal...")
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, len(rates))
        if rates_m5 is not None:
            df_m5 = pd.DataFrame(rates_m5)
            df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
            df_m5.to_csv(os.path.join(output_dir, f"XAUUSD_M5_MASTER_REAL_DATA.csv"), index=False)
            print(f"📈 M5 Guardado: {len(df_m5):,} barras.")
    else:
        print("❌ El broker no ha enviado datos M1.")

    mt5.shutdown()

if __name__ == "__main__":
    extract_oro_puro_dataset()
