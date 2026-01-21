import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Permute, Multiply, Flatten, Input
from tensorflow.keras import backend as K

# --------------------------------------------------------------------------------
# 🦅 TITAN V3 ULTIMATE - "PLATINUM COMBO" ENGINE (FIXED & AUGMENTED)
# --------------------------------------------------------------------------------
# UPGRADES V2:
# 1. FIX: DATA LEAKAGE (Strict standard scaler fit on Train only).
# 2. FIX: FEATURE STARVATION (Added Log Volume + Volatility).
# 3. CORE: Focal Loss + Triple Barrier preserved.
# --------------------------------------------------------------------------------

print("🦅 INICIANDO MOTOR 'ULTIMATE V2' (PLATINUM + LEAKAGE FIX)...")

# 1. ORQUESTACIÓN DE RUTAS (SMART PATHS - RICK FIX)
# --------------------------------------------------------------------------------
def find_project_root(start_path, target_folder="AchillesTraining"):
    current = os.path.abspath(start_path)
    while True:
        head, tail = os.path.split(current)
        if tail == target_folder: return current
        if head == current: break
        current = head
        
    if os.path.exists(r'c:\Users\David\AchillesTraining'): return r'c:\Users\David\AchillesTraining'
    if os.path.exists('/content/drive/MyDrive/AchillesTraining'): return '/content/drive/MyDrive/AchillesTraining'
    
    sync_paths = [
        '/content/drive/Othercomputers/Mi portátil/AchillesTraining',
        '/content/drive/Othercomputers/My Laptop/AchillesTraining',
        '/content/drive/Ordenadores/Mi portátil/AchillesTraining'
    ]
    for p in sync_paths:
        if os.path.exists(p): return p

    if os.path.exists('/content/drive'):
        for root, dirs, files in os.walk('/content/drive'):
            if target_folder in dirs: return os.path.join(root, target_folder)
    return None

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("☁️ ENTORNO COLAB DETECTADO. MONTANDO DRIVE...")
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SEARCH_START = '/content/drive'
else:
    print("💻 ENTORNO LOCAL DETECTADO.")
    try: SEARCH_START = os.path.dirname(os.path.abspath(__file__))
    except NameError: SEARCH_START = os.getcwd()

PROJECT_ROOT = find_project_root(SEARCH_START, 'AchillesTraining')
if not PROJECT_ROOT:
    sys.exit("❌ ERROR CRÍTICO: No se encontró 'AchillesTraining'.")

print(f"✅ PROYECTO: {PROJECT_ROOT}")
os.chdir(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, '00_FACTORY', 'TITAN_V3', 'output', 'v4.5_ULTIMATE')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --------------------------------------------------------------------------------
# 2. FOCAL LOSS & TRIPLE BARRIER
# --------------------------------------------------------------------------------
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss adaptada a Keras."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        return K.sum(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

def apply_triple_barrier(prices, volatility, barrier_width=1.5, horizon=10):
    """Etiquetado dinámico basado en volatilidad."""
    labels = []
    for i in range(len(prices) - horizon):
        current_p = prices[i]
        vol = volatility[i]
        if vol == 0: vol = 0.0001
        
        upper = current_p * (1 + barrier_width * vol)
        lower = current_p * (1 - barrier_width * vol)
        future_window = prices[i+1 : i+horizon+1]
        
        label = 2 # HOLD
        hit_upper = np.any(future_window >= upper)
        hit_lower = np.any(future_window <= lower)
        
        if hit_upper and not hit_lower: label = 0 # BUY
        elif hit_lower and not hit_upper: label = 1 # SELL
        elif hit_upper and hit_lower:
            first_upper = np.argmax(future_window >= upper)
            first_lower = np.argmax(future_window <= lower)
            if first_upper < first_lower: label = 0
            else: label = 1
        labels.append(label)
    return np.array(labels)

# --------------------------------------------------------------------------------
# 3. CARGA Y FEATURE ENGINEERING (AUGMENTED)
# --------------------------------------------------------------------------------
FILENAME = "XAUUSD_M5_2020-2025_Execution.csv"
FILEPATH = os.path.join(DATA_PATH, FILENAME)

if not os.path.exists(FILEPATH):
    glob_search = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PROJECT_ROOT) for f in filenames if f == FILENAME]
    if glob_search: FILEPATH = glob_search[0]
    else: sys.exit(f"❌ ERROR: No encuentro {FILENAME}")

print(f"🦅 Loading {FILENAME}...")
df = pd.read_csv(FILEPATH)
df.columns = [c.strip().lower() for c in df.columns]

# --- 3-CHANNEL INPUT ---
print("⚗️ CALCULATING FEATURES (3-DIMENSIONS)...")
# 1. Log Returns (Price Action)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
# 2. Log Volume (Liquidity) - Se suma 1 para evitar log(0)
df['log_vol'] = np.log1p(df['tick_volume']) 
# 3. Rolling Volatility (Regime)
df['volatility'] = df['log_ret'].rolling(window=50).std()

df.dropna(inplace=True)

# --- ETIQUETADO ---
print("🎯 Applying Triple Barrier...")
prices = df['close'].values
volatility = df['volatility'].values
labels = apply_triple_barrier(prices, volatility, barrier_width=1.5, horizon=12)

# Trim DF to Labels
df = df.iloc[:len(labels)]
y_ints = labels
y_categorical = tf.keras.utils.to_categorical(y_ints, num_classes=3)

print("⚖️ Class Distribution:")
unique, counts = np.unique(y_ints, return_counts=True)
print(dict(zip(unique, counts)))

# Class Weights (x2 boost for Buy/Sell)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict[0] *= 2.0; class_weights_dict[1] *= 2.0
print(f"⚖️ Adjusted Weights: {class_weights_dict}")

# --------------------------------------------------------------------------------
# 4. PREPARACIÓN DE SECUENCIAS (CON STRICT SPLIT & SCALER FIX)
# --------------------------------------------------------------------------------
SEQ_LEN = 60
# Seleccionar features para input
features = df[['log_ret', 'log_vol', 'volatility']].values

# SPLIT INDEX (80% Train, 20% Val) - SIN ALEATORIEDAD (Time Series)
split_idx = int(len(features) * 0.8)

train_data_raw = features[:split_idx]
val_data_raw = features[split_idx:]
train_labels_raw = y_categorical[:split_idx]
val_labels_raw = y_categorical[split_idx:]

print(f"🔪 SPLIT: Train={len(train_data_raw)}, Val={len(val_data_raw)}")

# SCALER FIT ONLY ON TRAIN (NO LEAKAGE!)
print("🛡️ FITTING SCALER ON TRAIN DATA ONLY...")
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_raw)
val_data_scaled = scaler.transform(val_data_raw) # Transform normaliza con media de Train

def create_sequences(data, labels, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(labels[i]) # Label del último paso (Triple Barrier ya mira a futuro)
    return np.array(X), np.array(y)

print("Tx Generating Sequences...")
X_train, y_train = create_sequences(train_data_scaled, train_labels_raw, SEQ_LEN)
X_val, y_val = create_sequences(val_data_scaled, val_labels_raw, SEQ_LEN)

print(f"✅ FINAL SHAPES: X_train={X_train.shape}, X_val={X_val.shape}")

# --------------------------------------------------------------------------------
# 5. MODELO TITAN V3 (INPUT LAYERS UPDATED)
# --------------------------------------------------------------------------------
def build_ultimate_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Bi-LSTM con más capacidad para 3 features
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Attention
    a = Permute((2, 1))(x)
    a = Dense(input_shape[0], activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    attention_mul = Multiply()([x, a_probs])
    x = Flatten()(attention_mul)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=categorical_focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model

# Input shape ahora es (60, 3)
model = build_ultimate_model((SEQ_LEN, 3))
print("🤖 MODELO COMPILADO (Input 3 Channels).")

# 6. ENTRENAMIENTO
# --------------------------------------------------------------------------------
print("🚀 INICIANDO ENTRENAMIENTO...")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_PATH, 'best_achilles_ultimate.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val), # VALIDACIÓN EXPLÍCITA SIN FUGAS
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights_dict,
    verbose=1
)

joblib.dump(scaler, os.path.join(OUTPUT_PATH, 'achilles_scaler_ultimate.pkl'))
print("💾 SCALER GUARDADO (FIT SOBRE TRAIN).")
print("🏆 OPERACIÓN PLATINUM V2 COMPLETADA.")
