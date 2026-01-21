import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import class_weight
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Permute, Multiply, Flatten, Input, Concatenate, LayerNormalization, SpatialDropout1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K

# [INJECTED_HELPER_FUNCTIONS]
def winsorize_and_scale(df, columns):
    """
    DIRTY HACK V1: Variable #40 (RobustScaler) + #51 (Winsorization)
    Objetivo: Eliminar el ruido de los ATH (All Time Highs) y eventos de cisne negro
    que están cegando al modelo con valores > 3 desviaciones estándar.
    """
    # Usamos RobustScaler que escala basado en percentiles (IQR), no en media/varianza
    scaler = RobustScaler()
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            # 1. WINSORIZATION (Clipping al 1% y 99%)
            upper_limit = df_copy[col].quantile(0.99)
            lower_limit = df_copy[col].quantile(0.01)
            df_copy[col] = np.clip(df_copy[col], lower_limit, upper_limit)
    
    # 2. ROBUST SCALING
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    return df_copy, scaler

# --------------------------------------------------------------------------------
# 🦅 TITAN V3 ULTIMATE V3 - "THE SURGICAL STRIKE"
# --------------------------------------------------------------------------------
# UPGRADES V3 (CRITICAL FIXES):
# 1. 🔭 BICÉFALO: Dual-Stream Input (M5 Sequence + D1 Context).
# 2. 🌊 RVOL: Volume Broker-Agnostic (Relative Volume).
# 3. 🛡️ ADAPTIVE BARRIER: Dynamic k based on Volatility Regime.
# 4. 🎚️ OUTPUT BIAS: Forced bravery injection.
# --------------------------------------------------------------------------------

print("🦅 INICIANDO MOTOR 'ULTIMATE V3' (SURGICAL FIX)...")

# 1. PATH SETUP
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
    return None

try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SEARCH_START = '/content/drive'
except ImportError:
    IN_COLAB = False
    SEARCH_START = os.getcwd()

PROJECT_ROOT = find_project_root(SEARCH_START, 'AchillesTraining')
if not PROJECT_ROOT: sys.exit("❌ ERROR CRÍTICO: No se encontró 'AchillesTraining'.")

os.chdir(PROJECT_ROOT)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, '00_FACTORY', 'TITAN_V3', 'output', 'v5_SURGICAL')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 2. FOCAL LOSS (PRESERVED)
# --------------------------------------------------------------------------------
# 2. FOCAL LOSS V2 (GODMODE - DIRTY HACK V1)
# --------------------------------------------------------------------------------
def focal_loss_titan_v3(gamma=2.5, alpha=[2.5, 2.5, 0.5]):
    """
    DIRTY HACK V1: Variable #123 (Focal Loss Recalibrada) - FIXED SHAPES & MAPPING
    Gamma 2.5: Enfoque extremo en ejemplos difíciles.
    Alpha [2.5, 2.5, 0.5]: 
        - Buy (Clase 0) -> 2.5 (GoldPriority)
        - Sell (Clase 1) -> 2.5 (GoldPriority)
        - Hold (Clase 2) -> 0.5 (LowPriority)
    Note: Code generates labels as 0=Buy, 1=Sell, 2=Hold. Protocol assumed 0=Hold. Fixed here.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Cross Entropy estándar
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Factor de peso Alpha (Balanceo de clases)
        # [FIX]: Convertir a tensor para broadcasting correcto [3] -> [Batch, 3]
        alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
        
        # Factor Focal (1 - p)^gamma
        focal_factor = tf.pow(1.0 - y_pred, gamma)
        
        # Combinación Element-wise (Sin reducir dimensiones prematuramente)
        loss_val = alpha_t * focal_factor * cross_entropy
        
        # Sumar sobre el eje de clases al final -> [Batch]
        return tf.reduce_sum(loss_val, axis=-1)
    
    return loss

# 3. ADAPTIVE TRIPLE BARRIER (UPGRADE #3)
# --------------------------------------------------------------------------------
def apply_adaptive_triple_barrier(prices, volatility, base_width=1.5, horizon=10):
    """
    Auto-Tuning Barrier:
    - Si Volatility Ratio > 1.2 (Mercado Loco) -> Amplía Barrera (k * 1.5)
    - Si Volatility Ratio < 0.8 (Mercado Muerto) -> Reduce Barrera (k * 0.8)
    """
    labels = []
    
    # Calcular Volatility Ratio (Vol Actual / Media Historica reciente)
    vol_series = pd.Series(volatility)
    vol_mean = vol_series.rolling(window=500).mean().fillna(method='bfill') # 500 barras ~ 2 dias
    vol_ratio = vol_series / vol_mean
    
    prices_arr = np.array(prices)
    vol_arr = np.array(volatility)
    ratio_arr = np.array(vol_ratio)

    for i in range(len(prices_arr) - horizon):
        current_p = prices_arr[i]
        vol = vol_arr[i]
        ratio = ratio_arr[i]
        
        # ADAPTIVE LOGIC
        if ratio > 1.5: adaptive_mult = base_width * 1.5
        elif ratio < 0.7: adaptive_mult = base_width * 0.7
        else: adaptive_mult = base_width
        
        width = vol * adaptive_mult
        upper = current_p * (1 + width)
        lower = current_p * (1 - width)
        
        future_window = prices_arr[i+1 : i+horizon+1]
        
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

# 4. DATA LOADING & FEATURE ENGINEERING (SURGICAL)
# --------------------------------------------------------------------------------
# Cargar Datos M5
FILENAME = "XAUUSD_M5_2020-2025_Execution.csv"
FILEPATH = os.path.join(DATA_PATH, FILENAME)
if not os.path.exists(FILEPATH):
    # Try finding recursively
    found = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PROJECT_ROOT) for f in filenames if f == FILENAME]
    if found: FILEPATH = found[0]
    else: sys.exit(f"❌ ERROR: No encuentro {FILENAME}")

print(f"🦅 Loading {FILENAME}...")
df_m5 = pd.read_csv(FILEPATH)
df_m5.columns = [c.strip().lower() for c in df_m5.columns]
df_m5['time'] = pd.to_datetime(df_m5['time'])
df_m5.set_index('time', inplace=True)

# --- RVOL CALCULATION (UPGRADE #2) ---
print("🌊 CALCULATING RVOL (Broker-Agnostic Volume)...")
# Agrupar por (Hora, Minuto) para tener el perfil de volumen intradía
df_m5['time_idx'] = df_m5.index.hour * 60 + df_m5.index.minute
vol_profile = df_m5.groupby('time_idx')['tick_volume'].transform('median')
df_m5['rvol'] = df_m5['tick_volume'] / (vol_profile + 1e-5)
# Clip extreme RVOL to prevent outliers
df_m5['rvol'] = df_m5['rvol'].clip(0, 5.0)

# --- FEATURES M5 ---
df_m5['log_ret'] = np.log(df_m5['close'] / df_m5['close'].shift(1))
df_m5['volatility'] = df_m5['log_ret'].rolling(window=12).std() # 1 hora
df_m5.dropna(inplace=True)

# --- CONTEXTO D1 (UPGRADE #1 - BICÉFALO) ---
print("🔭 GENERATING D1 CONTEXT LAYERS...")
# Resample to D1
df_d1 = df_m5['close'].resample('1D').agg(['last', 'max', 'min']).dropna()
df_d1['d1_ret'] = np.log(df_d1['last'] / df_d1['last'].shift(1))
df_d1['d1_range'] = (df_d1['max'] - df_d1['min']) / df_d1['last']
df_d1['d1_vol'] = df_d1['d1_ret'].rolling(5).std()
df_d1.fillna(0, inplace=True)

# Merge D1 back to M5 (Forward Fill)
# Cada vela M5 sabrá "cómo cerró el día de ayer" (Contexto pasado, no futuro)
df_m5['date'] = df_m5.index.normalize() - pd.Timedelta(days=1) # Join con el día ANTERIOR cerrado
df_merged = df_m5.merge(df_d1[['d1_ret', 'd1_range', 'd1_vol']], left_on='date', right_index=True, how='left').fillna(0)

# Main Features
features_m5 = df_merged[['log_ret', 'rvol', 'volatility']].values
features_d1 = df_merged[['d1_ret', 'd1_range', 'd1_vol']].values # Contexto repetido

# --- LABELS (ADAPTIVE) ---
# --- LABELS (ADAPTIVE - WIDER HORIZON) ---
print("🛡️ APPLYING ADAPTIVE TRIPLE BARRIER (WIDER HORIZON 2H)...")
prices = df_merged['close'].values
volatility_for_labels = df_merged['volatility'].values
# HORIZON UPDATE: 12 -> 24 (2 Horas para filtrar ruido Browniano)
labels = apply_adaptive_triple_barrier(prices, volatility_for_labels, base_width=1.5, horizon=24)

# Trim Data
valid_len = len(labels)
features_m5 = features_m5[:valid_len]
features_d1 = features_d1[:valid_len]
y_ints = labels
y_categorical = tf.keras.utils.to_categorical(y_ints, num_classes=3)

# 5. SEQUENCE GENERATION (DUAL STREAM)
# --------------------------------------------------------------------------------
SEQ_LEN = 60
split_idx = int(len(features_m5) * 0.8)

# SCALER ROBUSTO + WINSORIZATION (GODMODE)
print("⚖️ FITTING SCALERS (WINSORIZATION + ROBUST)...")

# Convert numpy arrays back to DF for winsorization helper
cols_m5 = ['log_ret', 'rvol', 'volatility']
cols_d1 = ['d1_ret', 'd1_range', 'd1_vol']

df_train_m5 = pd.DataFrame(features_m5[:split_idx], columns=cols_m5)
df_val_m5 = pd.DataFrame(features_m5[split_idx:], columns=cols_m5)

df_train_d1 = pd.DataFrame(features_d1[:split_idx], columns=cols_d1)
df_val_d1 = pd.DataFrame(features_d1[split_idx:], columns=cols_d1)

# Apply Winsorization + Robust Scaling (ONLY FIT ON TRAIN)
df_train_m5_scaled, scaler_m5 = winsorize_and_scale(df_train_m5, cols_m5)
# Transform Validation (No fit, use same scaler, but clip outliers first with same limits? No, winsorize checks its own distribution or predefined? 
# Standard practice: Fit scaler on train. Transform val. 
# Winsorization should ideally be done per-batch or on train only to remove outliers from learning.
# For simplicity and robustness: We transform Val using the fitted scaler. Winsorization on Val is debated, we will skip clipping on Val to see real outliers impact or clip using Train limits.
# Let's simplify: Just RobustScale Val. Winsorize Train to clean learning.
train_m5_norm = df_train_m5_scaled.values
val_m5_norm = scaler_m5.transform(df_val_m5) # RobustScaler handles outliers relatively well

df_train_d1_scaled, scaler_d1 = winsorize_and_scale(df_train_d1, cols_d1)
train_d1_norm = df_train_d1_scaled.values
val_d1_norm = scaler_d1.transform(df_val_d1)

def create_dual_sequences(data_m5, data_d1, labels, seq_len):
    X_m5, X_d1, y = [], [], []
    for i in range(seq_len, len(data_m5)):
        X_m5.append(data_m5[i-seq_len:i])
        # Para D1, tomamos el valor actual (static context approach simple para la rama densa)
        # Ojo: data_d1 ya tiene ffill, así que data_d1[i] es el contexto del día anterior
        X_d1.append(data_d1[i]) 
        y.append(labels[i])
    return np.array(X_m5), np.array(X_d1), np.array(y)

print("Tx Generating Dual Sequences...")
X_train_m5, X_train_d1, y_train = create_dual_sequences(train_m5_norm, train_d1_norm, y_categorical[:split_idx], SEQ_LEN)
X_val_m5, X_val_d1, y_val = create_dual_sequences(val_m5_norm, val_d1_norm, y_categorical[split_idx:], SEQ_LEN)

# 6. MODELO BICÉFALO (UPGRADE #1 & #4)
# --------------------------------------------------------------------------------
def build_surgical_model(input_shape_m5, input_shape_d1):
    # --- RAMA 1: M5 (Rápida) ---
    in_m5 = Input(shape=input_shape_m5, name='Input_M5')
    
    # [GODMODE] SPATIAL DROPOUT: Apagar canales enteros
    x = SpatialDropout1D(0.3)(in_m5)
    
    # [GODMODE] L1/L2 REGULARIZATION + REDUCED CAPACITY (128 -> 64)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(x)
    x = Dropout(0.4)(x) # More aggressive dropout
    
    x = LSTM(32)(x) # 64 -> 32
    x = BatchNormalization()(x)
    
    # --- RAMA 2: D1 (Contexto) ---
    in_d1 = Input(shape=input_shape_d1, name='Input_D1') 
    y = Dense(16, activation='relu')(in_d1) # 32 -> 16
    y = BatchNormalization()(y)
    
    # --- FUSIÓN ---
    concat = Concatenate()([x, y])
    z = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(concat) # 64 -> 32
    
    # [GODMODE] DROPOUT AGRESIVO FINAL
    z = Dropout(0.5)(z)
    
    # --- OUTPUT BIAS INJECTION (UPGRADE #4) ---
    # Forzamos sesgo inicial para Buy(0) y Sell(1) vs Hold(2)
    # [Buy_Bias, Sell_Bias, Hold_Bias] -> [0.5, 0.5, -1.0] para animarlo
    output_bias = tf.keras.initializers.Constant([0.2, 0.2, -0.5]) 
    
    out = Dense(3, activation='softmax', bias_initializer=output_bias)(z)
    
    model = Model(inputs=[in_m5, in_d1], outputs=out)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4) # AdamW FTW
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss_titan_v3(), # [GODMODE] CUSTOM FOCAL LOSS V2
        metrics=['accuracy']
    )
    return model

model = build_surgical_model((SEQ_LEN, 3), (3,)) # 3 features M5, 3 features D1
print("🤖 MODELO QUIRÚRGICO M5+D1 COMPILADO.")

# 7. TRAINING
# --------------------------------------------------------------------------------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_PATH, 'titan_surgical_v3.keras'),
    monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Class Weights (Sigue siendo útil como refuerzo)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)
cw_dict = dict(enumerate(cw))
cw_dict[0] *= 1.5; cw_dict[1] *= 1.5 # Boost extra

print("🚀 INICIANDO ENTRENAMIENTO...")
history = model.fit(
    [X_train_m5, X_train_d1], y_train,
    epochs=50,
    batch_size=64, # Batch más grande para estabilidad
    validation_data=([X_val_m5, X_val_d1], y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
    # class_weight=cw_dict, # [GODMODE FIX]: Disabled to avoid Double-Penalty with Focal Loss Alpha
    verbose=1
)

joblib.dump(scaler_m5, os.path.join(OUTPUT_PATH, 'scaler_m5.pkl'))
joblib.dump(scaler_d1, os.path.join(OUTPUT_PATH, 'scaler_d1.pkl'))
print("🏆 OPERACIÓN QUIRÚRGICA COMPLETADA.")
