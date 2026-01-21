import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Permute, Multiply, Flatten, Input
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW

# --------------------------------------------------------------------------------
# 🛡️ ACHILLES TITAN V3 - TRAINING ENGINE (RICK PROTOCOL)
# --------------------------------------------------------------------------------

print("🦅 INICIANDO MOTOR DE ENTRENAMIENTO TITAN V3 (PYTHON PURE)...")

# 1. ORQUESTACIÓN DE RUTAS (SMART PATHS)
# --------------------------------------------------------------------------------
# 1. ORQUESTACIÓN DE RUTAS (SMART PATHS - RICK FIX)
# --------------------------------------------------------------------------------
def find_project_root(start_path, target_folder="AchillesTraining"):
    # Estrategia 1: Escalar hacia arriba ( Parents )
    # Si estamos en .../AchillesTraining/00_FACTORY/TITAN_V3
    # Queremos encontrar .../AchillesTraining
    current = os.path.abspath(start_path)
    while True:
        head, tail = os.path.split(current)
        if tail == target_folder:
            return current
        if head == current: # Raíz del sistema alcanzada
            break
        current = head
        
    # Estrategia 2: Búsqueda explícita (Fail-safes conocidos)
    # Windows Local
    if os.path.exists(r'c:\Users\David\AchillesTraining'):
        return r'c:\Users\David\AchillesTraining'
        
    # Colab Standard (MyDrive)
    if os.path.exists('/content/drive/MyDrive/AchillesTraining'):
        return '/content/drive/MyDrive/AchillesTraining'
        
    # Colab "Ordenadores" (Drive for Desktop Sync)
    # El usuario tiene "Ordenadores > Mi portátil > AchillesTraining"
    possible_sync_paths = [
        '/content/drive/Othercomputers/Mi portátil/AchillesTraining',
        '/content/drive/Othercomputers/My Laptop/AchillesTraining',
        '/content/drive/Ordenadores/Mi portátil/AchillesTraining'
    ]
    for p in possible_sync_paths:
        if os.path.exists(p):
            print(f"🔭 Detectada ruta de Sincronización Desktop: {p}")
            return p

    # Estrategia 3: Búsqueda Profunda en /content/drive (Si falla lo anterior)
    if os.path.exists('/content/drive'):
        print("🕵️‍♂️ Iniciando búsqueda profunda en Drive (esto puede tardar unos segundos)...")
        for root, dirs, files in os.walk('/content/drive'):
            if target_folder in dirs:
                return os.path.join(root, target_folder)
                
    return None

# Detectar entorno (Robustez RICK V2)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("☁️ DETECTADO ENTORNO COLAB. MONTANDO DRIVE...")
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    # Una vez montado, definimos el punto de partida
    # Incluimos lógica para 'Otros Ordenadores' dentro de find_project_root, 
    # pero empezamos la busqueda desde /content/drive
    SEARCH_START = '/content/drive' 
else:
    print("💻 DETECTADO ENTORNO LOCAL (WINDOWS/LINUX).")
    # Usamos la ubicación del script actual como ancla
    try:
        SEARCH_START = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        print("⚠️ Entorno Interactivo detectado (__file__ not found). Usando CWD.")
        SEARCH_START = os.getcwd()

PROJECT_ROOT = find_project_root(SEARCH_START, 'AchillesTraining')

if not PROJECT_ROOT:
    print("❌ ERROR CRÍTICO: No se encontró la carpeta 'AchillesTraining'.")
    print(f"   -> Buscando desde: {SEARCH_START}")
    sys.exit(1)
else:
    print(f"✅ PROYECTO LOCALIZADO EN: {PROJECT_ROOT}")

# --------------------------------------------------------------------------------
# 2. DEFINICIÓN DE ENTORNO DE TRABAJO (CRÍTICO)
# --------------------------------------------------------------------------------
# MOVERSE A LA RAIZ DEL PROYECTO
# Esto soluciona el problema de las rutas relativas en Colab vs Local
print(f"📂 CAMBIANDO DIRECTORIO DE TRABAJO A: {PROJECT_ROOT}")
os.chdir(PROJECT_ROOT)
print(f"📍 CWD ACTUAL: {os.getcwd()}")

# --------------------------------------------------------------------------------
# 2. DEFINICIÓN DE ENTORNO DE TRABAJO (CRÍTICO)
# --------------------------------------------------------------------------------
# MOVERSE A LA RAIZ DEL PROYECTO
print(f"📂 CAMBIANDO DIRECTORIO DE TRABAJO A: {PROJECT_ROOT}")
os.chdir(PROJECT_ROOT)
print(f"📍 CWD ACTUAL: {os.getcwd()}")

# CORRECCIÓN DE RUTA BASADA EN EVIDENCIA (LOGS)
# Los archivos están en AchillesTraining/data, NO en 00_FACTORY...
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, '00_FACTORY', 'TITAN_V3', 'output', 'v4.5')
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"✅ DATA PATH (CORREGIDO): {DATA_PATH}")

# Listar contenido de Data para debug (RICK EVIDENCE)
if os.path.exists(DATA_PATH):
    print("📋 CONTENIDO DE LA CARPETA DATA:")
    for f in os.listdir(DATA_PATH):
        if f.endswith('.csv'):
            print(f"   - {f}")
else:
    print(f"❌ ERROR: La carpeta 'data' no existe en: {DATA_PATH}")

# --------------------------------------------------------------------------------
# 3. MANIFIESTO DE DATOS (RICK PRECISION MODE)
# --------------------------------------------------------------------------------
# Actualizado con los nombres REALES vistos en el log

DATA_FILES_CONFIG = [
    # --- CAPA 1: ROBUSTEZ (Crisis D1 - "The Survivors") ---
    {"file": "XAUUSD_D1_2000-2009_DotCom-Lehman.csv", "type": "CRISIS_D1"},
    {"file": "XAUUSD_D1_2011-2012_Euro.csv", "type": "CRISIS_D1"},
    {"file": "XAUUSD_D1_2020_COVID.csv", "type": "CRISIS_D1"},
    {"file": "XAUUSD_D1_2022_Ukraine.csv", "type": "CRISIS_D1"},
    
    # --- CAPA 2: EJECUCIÓN (Oro Puro M5 - "The Sniper") ---
    # CORRECCIÓN: Usamos el archivo que REALMENTE existe en tu carpeta
    {"file": "XAUUSD_M5_2020-2025_Execution.csv", "type": "EXECUTION_M5"} 
]

dfs = []
for item in DATA_FILES_CONFIG:
    filename = item['file']
    datatype = item['type']
    
    # RUTA EXACTA Y RELATIVA (Ya estamos en PROJECT_ROOT)
    filepath = os.path.join(DATA_PATH, filename)
    
    print(f"🦅 Loading [{datatype}] {filename}...")
    if not os.path.exists(filepath):
        print(f"❌ ERROR: El archivo NO está en: {filepath}")
        print(f"   (Buscando en {DATA_PATH})")
        continue
        
    try:
        d = pd.read_csv(filepath)
        d.columns = [c.strip().lower() for c in d.columns]
        
        # Validar columnas
        if 'close' in d.columns:
            dfs.append(d)
        else:
            print(f"   ⚠️ Skipped {filename} (No 'close' column)")
    except Exception as e:
        print(f"   ❌ Error reading {filename}: {e}")

if not dfs:
    print("❌ ERROR CRÍTICO: No se ha cargado NINGÚN archivo CSV.")
    print("   -> Verifica que has subido la carpeta '00_FACTORY/TITAN_V3/data' o 'TEMPORAL' al Drive.")
    print("   -> Debug: Listando contenido de PROJECT_ROOT para que veas qué hay:")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        level = root.replace(PROJECT_ROOT, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f.endswith('.csv'):
                print(f"{subindent}{f}")
    raise ValueError("CRITICAL: No valid CSV data loaded! See tree above.")

if not dfs:
    raise ValueError("CRITICAL: No valid CSV data loaded! Check paths.")

full_df = pd.concat(dfs, ignore_index=True)
full_df.dropna(subset=['close'], inplace=True)
print(f"🔥 TOTAL TRAINING ROWS (FUSED): {len(full_df)}")

# 4. PREPROCESAMIENTO (PIPELINE)
# --------------------------------------------------------------------------------
print("⚗️ Normalizando Datos (StandardScaler)...")
data = full_df['close'].values.reshape(-1, 1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

SEQ_LEN = 60
X, y = [], []

print("⚗️ Generating Sequences via Sliding Window...")
# Optimización: Usar stride_tricks si fuera necesario, pero mantendremos bucle simple por claridad en log
for i in range(SEQ_LEN, len(data_scaled) - 5):
    X.append(data_scaled[i-SEQ_LEN:i])
    
    # Labeling Logic (Future Horizon = 5 bars)
    # NOTA: En D1, 5 barras es una semana. En M5, son 25 minutos. 
    # Esta heterogeneidad es aceptable porque buscamos patrones de "movimiento %", no tiempo absoluto.
    future_price = data[i+5][0]
    current_price = data[i][0]
    
    if current_price == 0: pct_change = 0
    else: pct_change = (future_price - current_price) / current_price
    
    # 3-Class Classification (Threshold 0.5%)
    if pct_change > 0.005: label = [0, 1, 0] # BUY
    elif pct_change < -0.005: label = [0, 0, 1] # SELL
    else: label = [1, 0, 0] # HOLD
    
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"✅ Training Data Shape: {X.shape}")

# 5. MODELO TITAN V3 (Bi-LSTM + Attention)
# --------------------------------------------------------------------------------
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    return Multiply()([inputs, a_probs])

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    
    attention_mul = attention_block(lstm_out, input_shape[0])
    attention_mul = Flatten()(attention_mul)
    
    dense = Dense(64, activation='relu')(attention_mul)
    dense = Dropout(0.2)(dense)
    outputs = Dense(3, activation='softmax')(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = AdamW(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((SEQ_LEN, 1))
print("🤖 ACHILLES V3 ARCHITECTURE: READY.")

# 6. ENTRENAMIENTO
# --------------------------------------------------------------------------------
print("🚀 STARTING TRAINING (3-PRO MODE)...")
callback_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_PATH, 'best_achilles_v3.keras'),
    monitor='accuracy',
    save_best_only=True
)

history = model.fit(
    X, y, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[callback_stop, callback_checkpoint]
)

# 7. GUARDADO DE ARTEFACTOS
# --------------------------------------------------------------------------------
joblib.dump(scaler, os.path.join(OUTPUT_PATH, 'achilles_scaler.pkl'))
print(f"💾 SCALER SAVED: {os.path.join(OUTPUT_PATH, 'achilles_scaler.pkl')}")
print(f"💾 MODEL SAVED: {os.path.join(OUTPUT_PATH, 'best_achilles_v3.keras')}")
print("🏆 MISSION COMPLETE.")
