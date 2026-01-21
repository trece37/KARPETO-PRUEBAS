
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
from scipy.stats import entropy
# Antropy dependency check disabled for portability, implementing manual Higuchi or simple check
# import antropy as ent 

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

# [PHASE 1 INJECTION] HIGUCHI FRACTAL DIMENSION (Optimized NumPy Implementation)
def higuchi_fd(x, kmax):
    """
    Calculate Higuchi Fractal Dimension.
    x: Time series (numpy array)
    kmax: scale parameter
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.array(range(1, kmax + 1))
    y_reg = np.empty(kmax)
    
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            indices = np.arange(m, n_times, k)
            if indices.size < 2:
                lm[m] = 0
                continue
            
            # Sum of absolute differences
            diffs = np.abs(x[indices[1:]] - x[indices[:-1]])
            Lmk = (np.sum(diffs) * (n_times - 1) / (indices.size * k)) / k
            lm[m] = Lmk
            
        lk[k - 1] = np.mean(lm)
        y_reg[k - 1] = np.log(lk[k - 1] if lk[k-1] > 0 else 1e-10)
        
    x_reg = np.log(1.0 / x_reg)
    # Slope of linear regression
    # Check if x_reg or y_reg contains Infs or NaNs
    slope, intercept = np.polyfit(x_reg, y_reg, 1)
    return slope

def calculate_causality_features(df, window_entropy=20, window_fractal=60):
    """
    PHASE 1 FEATURE INJECTION:
    1. Shannon Entropy Z-Score (Chaos Detector)
    2. Higuchi Fractal Dimension (Roughness Filter)
    3. Order Flow Imbalance (OFI Proxy)
    """
    print("🔬 GENERATING CAUSALITY FEATURES (PHASE 1)...")
    df_feat = df.copy()
    
    # 1. Shannon Entropy Z-Score
    def get_rolling_entropy(series):
        # Discretize into 10 bins
        try:
            counts = pd.cut(series, bins=10).value_counts(normalize=True)
            return entropy(counts, base=2)
        except:
            return 0.0

    df_feat['returns'] = df_feat['close'].pct_change().fillna(0)
    
    # Rolling Apply is slow, optimized approach:
    # Small window entropy can be approximate or cached? standard rolling apply is fine for offline training
    print("   > Calculating Entropy (This may take a moment)...")
    df_feat['entropy_raw'] = df_feat['returns'].rolling(window=window_entropy).apply(get_rolling_entropy, raw=False)
    
    # Z-Score Normalization (Historical Context)
    roll_mean = df_feat['entropy_raw'].rolling(252).mean()
    roll_std = df_feat['entropy_raw'].rolling(252).std()
    df_feat['entropy_z'] = (df_feat['entropy_raw'] - roll_mean) / (roll_std + 1e-6)
    
    # 2. Higuchi Fractal Dimension (HFD)
    print("   > Calculating Fractal Dimension...")
    # Using window_fractal (60 bars) for HFD calculation
    # Vectorizing is hard, using rolling apply with numpy optimized func
    df_feat['fractal_higuchi'] = df_feat['close'].rolling(window=window_fractal).apply(
        lambda x: higuchi_fd(x.values, kmax=5) if len(x) == window_fractal else 1.5, raw=False
    )
    
    # 3. Order Flow Imbalance (NOFI)
    print("   > Calculating OFI Proxy...")
    price_diff = df_feat['close'].diff()
    # Tick Rule: +1 if price up, -1 if price down, else previous
    tick_dir = np.sign(price_diff)
    tick_dir = tick_dir.replace(0, method='ffill').fillna(0)
    
    df_feat['ofi_raw'] = tick_dir * df_feat['tick_volume']
    df_feat['nofi'] = df_feat['ofi_raw'] / (df_feat['tick_volume'] + 1e-6)
    
    # Cleanup Nans
    df_feat.fillna(method='bfill', inplace=True)
    df_feat.fillna(0, inplace=True)
    
    return df_feat

# --------------------------------------------------------------------------------
# 🦅 TITAN V4 PHASE 1 - "CAUSALITY INJECTION"
# --------------------------------------------------------------------------------

print("🦅 INICIANDO MOTOR 'TITAN V4 PHASE 1' (POLY-FOCAL + FRACTAL)...")

# 1. PATH SETUP
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
OUTPUT_PATH = os.path.join(PROJECT_ROOT, '00_FACTORY', 'TITAN_V3', 'output', 'v4_PHASE1')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 2. POLY-FOCAL LOSS V3 (PHASE 1 UPGRADE)
# --------------------------------------------------------------------------------
def poly_focal_loss_v3(gamma=2.5, alpha=[2.5, 2.5, 0.5], epsilon=1.0):
    """
    [PHASE 1] Poly-Focal Loss (Taylor Expansion Refinement)
    Prevents model laziness when p_t is close to 1.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # 1. Base Focal Loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
        focal_factor = tf.pow(1.0 - y_pred, gamma)
        base_focal_loss = alpha_t * focal_factor * cross_entropy
        
        # 2. Poly Term (Polynomial Expansion)
        # Adds gradient support for difficult examples
        # Poly-1 term: epsilon * (1-pt)^(gamma+1)
        # Note: We need a scalar metric per class, but simplified implementation:
        
        # We calculate pt (probability of true class)
        # Since y_true is one-hot, pt = sum(y_true * y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True) 
        poly_term = epsilon * tf.pow(1.0 - pt, gamma + 1.0)
        
        final_loss = base_focal_loss + poly_term
        
        return tf.reduce_sum(final_loss, axis=-1)
    
    return loss

# 3. ADAPTIVE TRIPLE BARRIER (PRESERVED)
def apply_adaptive_triple_barrier(prices, volatility, base_width=1.5, horizon=10):
    labels = []
    vol_series = pd.Series(volatility)
    vol_mean = vol_series.rolling(window=500).mean().fillna(method='bfill')
    vol_ratio = vol_series / vol_mean
    
    prices_arr = np.array(prices)
    vol_arr = np.array(volatility)
    ratio_arr = np.array(vol_ratio)

    for i in range(len(prices_arr) - horizon):
        current_p = prices_arr[i]
        vol = vol_arr[i]
        ratio = ratio_arr[i]
        
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

# 4. DATA LOADING & FEATURE ENGINEERING
FILENAME = "XAUUSD_M5_2020-2025_Execution.csv"
FILEPATH = os.path.join(DATA_PATH, FILENAME)
if not os.path.exists(FILEPATH):
    found = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PROJECT_ROOT) for f in filenames if f == FILENAME]
    if found: FILEPATH = found[0]
    else: sys.exit(f"❌ ERROR: No encuentro {FILENAME}")

print(f"🦅 Loading {FILENAME}...")
df_m5 = pd.read_csv(FILEPATH)
df_m5.columns = [c.strip().lower() for c in df_m5.columns]
df_m5['time'] = pd.to_datetime(df_m5['time'])
df_m5.set_index('time', inplace=True)

# RVOL
df_m5['time_idx'] = df_m5.index.hour * 60 + df_m5.index.minute
vol_profile = df_m5.groupby('time_idx')['tick_volume'].transform('median')
df_m5['rvol'] = df_m5['tick_volume'] / (vol_profile + 1e-5)
df_m5['rvol'] = df_m5['rvol'].clip(0, 5.0)

# Base Features
df_m5['log_ret'] = np.log(df_m5['close'] / df_m5['close'].shift(1))
df_m5['volatility'] = df_m5['log_ret'].rolling(window=12).std()
df_m5.dropna(inplace=True)

# [PHASE 1] INJECT CAUSALITY FEATURES
df_m5 = calculate_causality_features(df_m5)

# Context D1
print("🔭 GENERATING D1 CONTEXT LAYERS...")
df_d1 = df_m5['close'].resample('1D').agg(['last', 'max', 'min']).dropna()
df_d1['d1_ret'] = np.log(df_d1['last'] / df_d1['last'].shift(1))
df_d1['d1_range'] = (df_d1['max'] - df_d1['min']) / df_d1['last']
df_d1['d1_vol'] = df_d1['d1_ret'].rolling(5).std()
df_d1.fillna(0, inplace=True)

df_m5['date'] = df_m5.index.normalize() - pd.Timedelta(days=1)
df_merged = df_m5.merge(df_d1[['d1_ret', 'd1_range', 'd1_vol']], left_on='date', right_index=True, how='left').fillna(0)

# Main Features [UPDATED LIST]
# Old: log_ret, rvol, volatility
# New: log_ret, rvol, volatility, entropy_z, fractal_higuchi, nofi
features_m5 = df_merged[['log_ret', 'rvol', 'volatility', 'entropy_z', 'fractal_higuchi', 'nofi']].values
features_d1 = df_merged[['d1_ret', 'd1_range', 'd1_vol']].values 

# Labels
print("🛡️ APPLYING ADAPTIVE TRIPLE BARRIER (WIDER HORIZON 2H)...")
prices = df_merged['close'].values
volatility_for_labels = df_merged['volatility'].values
labels = apply_adaptive_triple_barrier(prices, volatility_for_labels, base_width=1.5, horizon=24)

# Trim Data
valid_len = len(labels)
features_m5 = features_m5[:valid_len]
features_d1 = features_d1[:valid_len]
y_ints = labels
y_categorical = tf.keras.utils.to_categorical(y_ints, num_classes=3)

# 5. SEQUENCE GENERATION
SEQ_LEN = 60
split_idx = int(len(features_m5) * 0.8)

print("⚖️ FITTING SCALERS (WINSORIZATION + ROBUST)...")
# Update Columns
cols_m5 = ['log_ret', 'rvol', 'volatility', 'entropy_z', 'fractal_higuchi', 'nofi'] 
cols_d1 = ['d1_ret', 'd1_range', 'd1_vol']

df_train_m5 = pd.DataFrame(features_m5[:split_idx], columns=cols_m5)
df_val_m5 = pd.DataFrame(features_m5[split_idx:], columns=cols_m5)
df_train_d1 = pd.DataFrame(features_d1[:split_idx], columns=cols_d1)
df_val_d1 = pd.DataFrame(features_d1[split_idx:], columns=cols_d1)

df_train_m5_scaled, scaler_m5 = winsorize_and_scale(df_train_m5, cols_m5)
train_m5_norm = df_train_m5_scaled.values
val_m5_norm = scaler_m5.transform(df_val_m5)

df_train_d1_scaled, scaler_d1 = winsorize_and_scale(df_train_d1, cols_d1)
train_d1_norm = df_train_d1_scaled.values
val_d1_norm = scaler_d1.transform(df_val_d1)

def create_dual_sequences(data_m5, data_d1, labels, seq_len):
    X_m5, X_d1, y = [], [], []
    for i in range(seq_len, len(data_m5)):
        X_m5.append(data_m5[i-seq_len:i])
        X_d1.append(data_d1[i]) 
        y.append(labels[i])
    return np.array(X_m5), np.array(X_d1), np.array(y)

print("Tx Generating Dual Sequences...")
X_train_m5, X_train_d1, y_train = create_dual_sequences(train_m5_norm, train_d1_norm, y_categorical[:split_idx], SEQ_LEN)
X_val_m5, X_val_d1, y_val = create_dual_sequences(val_m5_norm, val_d1_norm, y_categorical[split_idx:], SEQ_LEN)

# 6. MODELO BICÉFALO (UPDATED INPUT SHAPE)
def build_surgical_model_v4(input_shape_m5, input_shape_d1):
    # --- RAMA 1: M5 (Rápida) ---
    in_m5 = Input(shape=input_shape_m5, name='Input_M5')
    
    # [PHASE 1] SPATIAL DROPOUT MAINTAINED
    x = SpatialDropout1D(0.3)(in_m5)
    
    # [PHASE 1] CAPACITY MAINTAINED (64 Neurons) - "Cognitive Minimalism"
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(x)
    x = Dropout(0.4)(x) 
    
    x = LSTM(32)(x) 
    x = BatchNormalization()(x)
    
    # --- RAMA 2: D1 (Contexto) ---
    in_d1 = Input(shape=input_shape_d1, name='Input_D1') 
    y = Dense(16, activation='relu')(in_d1) 
    y = BatchNormalization()(y)
    
    # --- FUSIÓN ---
    concat = Concatenate()([x, y])
    z = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(concat)
    
    z = Dropout(0.5)(z)
    
    # Output Bias
    output_bias = tf.keras.initializers.Constant([0.2, 0.2, -0.5]) 
    
    out = Dense(3, activation='softmax', bias_initializer=output_bias)(z)
    
    model = Model(inputs=[in_m5, in_d1], outputs=out)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4) 
    
    model.compile(
        optimizer=optimizer,
        loss=poly_focal_loss_v3(), # [PHASE 1] POLY-FOCAL LOSS
        metrics=['accuracy']
    )
    return model

# Update Input Shape: 6 Features M5 instead of 3
model = build_surgical_model_v4((SEQ_LEN, 6), (3,)) 
print("🤖 MODELO TITAN V4 PHASE 1 (POLY-FOCAL + 6 FEATURES) COMPILADO.")

# 7. TRAINING
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_PATH, 'titan_v4_phase1.keras'),
    monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("🚀 INICIANDO ENTRENAMIENTO PHASE 1...")
history = model.fit(
    [X_train_m5, X_train_d1], y_train,
    epochs=50,
    batch_size=64,
    validation_data=([X_val_m5, X_val_d1], y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

joblib.dump(scaler_m5, os.path.join(OUTPUT_PATH, 'scaler_m5.pkl'))
joblib.dump(scaler_d1, os.path.join(OUTPUT_PATH, 'scaler_d1.pkl'))
print("🏆 OPERACIÓN PHASE 1 COMPLETADA.")
