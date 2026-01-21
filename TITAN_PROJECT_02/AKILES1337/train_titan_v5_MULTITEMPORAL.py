
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

# [INJECTED_HELPER_FUNCTIONS]
def winsorize_and_scale(df, columns):
    """
    DIRTY HACK V1: Variable #40 (RobustScaler) + #51 (Winsorization)
    Objetivo: Eliminar el ruido de los ATH (All Time Highs) y eventos de cisne negro
    que están cegando al modelo con valores > 3 desviaciones estándar.
    """
    scaler = RobustScaler()
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            upper_limit = df_copy[col].quantile(0.99)
            lower_limit = df_copy[col].quantile(0.01)
            df_copy[col] = np.clip(df_copy[col], lower_limit, upper_limit)
    
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy, scaler

# [PHASE 1 INJECTION] HIGUCHI FRACTAL DIMENSION (Optimized NumPy Implementation)
def higuchi_fd(x, kmax):
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
            diffs = np.abs(x[indices[1:]] - x[indices[:-1]])
            Lmk = (np.sum(diffs) * (n_times - 1) / (indices.size * k)) / k
            lm[m] = Lmk
        lk[k - 1] = np.mean(lm)
        y_reg[k - 1] = np.log(lk[k - 1] if lk[k-1] > 0 else 1e-10)
        
    x_reg = np.log(1.0 / x_reg)
    slope, intercept = np.polyfit(x_reg, y_reg, 1)
    return slope

# --------------------------------------------------------------------------------
# 🦅 TITAN V5 - "MULTITEMPORAL FUSION" (DEEPSEEK ARCHITECTURE)
# --------------------------------------------------------------------------------

print("🦅 INICIANDO MOTOR 'TITAN V5 MULTITEMPORAL' (FUSIÓN DE CONTEXTOS)...")

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
OUTPUT_PATH = os.path.join(PROJECT_ROOT, '00_FACTORY', 'TITAN_V3', 'output', 'v5_DEEPSEEK')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 2. DATA LOADING & FEATURE ENGINEERING
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

# RVOL (V3 Feature - Maintained)
df_m5['time_idx'] = df_m5.index.hour * 60 + df_m5.index.minute
vol_profile = df_m5.groupby('time_idx')['tick_volume'].transform('median')
df_m5['rvol'] = df_m5['tick_volume'] / (vol_profile + 1e-5)
df_m5['rvol'] = df_m5['rvol'].clip(0, 5.0)

# Base Features (M5)
df_m5['log_ret'] = np.log(df_m5['close'] / df_m5['close'].shift(1))
df_m5['volatility'] = df_m5['log_ret'].rolling(window=12).std()

# Causality Features (V4 - Maintained but potentially noisy, we keep them as inputs)
print("🔬 CALCULATING V4 CAUSALITY FEATURES (Entropy, Fractal, OFI)...")
df_m5['returns_raw'] = df_m5['close'].pct_change().fillna(0)
# Entropy
def get_entropy(x):
    try:
        return entropy(pd.cut(x, 10).value_counts(normalize=True), base=2)
    except: return 0
df_m5['entropy'] = df_m5['returns_raw'].rolling(20).apply(get_entropy, raw=False)
# Z-Score Entropy
df_m5['entropy_z'] = (df_m5['entropy'] - df_m5['entropy'].rolling(252).mean()) / (df_m5['entropy'].rolling(252).std() + 1e-6)
# Fractal
df_m5['fractal_higuchi'] = df_m5['close'].rolling(60).apply(lambda x: higuchi_fd(x.values, 5) if len(x)==60 else 1.5, raw=False)
# OFI
price_diff = df_m5['close'].diff()
tick_dir = np.sign(price_diff).replace(0, method='ffill').fillna(0)
df_m5['nofi'] = (tick_dir * df_m5['tick_volume']) / (df_m5['tick_volume'] + 1e-6)

df_m5.dropna(inplace=True)

# --------------------------------------------------------------------------------
# MULTITEMPORAL FUSION (THE DEEPSEEK CORE)
# --------------------------------------------------------------------------------
print("🔭 GENERATING MULTITEMPORAL CONTEXT LAYERS (DeepSeek Specs)...")

def create_context_without_lookahead(df, period='1H', ma_window=20):
    """
    Resamples data to a higher timeframe and reindexes back to M5.
    CRITICAL: Uses .shift(1) to ensure M5 candle at 10:05 sees only H1 data UP TO 10:00.
    """
    # Resample to get Close of the higher timeframe
    df_resampled = df['close'].resample(period).last().dropna()
    # Calculate MA on the Higher Timeframe
    df_resampled_ma = df_resampled.rolling(window=ma_window).mean()
    # Reindex to M5 (Forward Fill: 10:00 value fills 10:05, 10:10...)
    # Shift(1): Moves the 10:00 value to start appearing at 10:05 (next bar), ensuring no peek-ahead.
    return df_resampled_ma.reindex(df.index, method='ffill').shift(1)

# Context Levels
df_m5['trend_h1'] = create_context_without_lookahead(df_m5, '1H', 20)
df_m5['trend_h4'] = create_context_without_lookahead(df_m5, '4H', 10)
df_m5['trend_d1'] = create_context_without_lookahead(df_m5, '1D', 5)

# Relative Positions (Regime Indicators)
df_m5['pos_vs_h1'] = df_m5['close'] / (df_m5['trend_h1'] + 1e-6)
df_m5['pos_vs_h4'] = df_m5['close'] / (df_m5['trend_h4'] + 1e-6)

# Volatility Context (ATR H4 Normalized)
df_h4_atr = df_m5['close'].resample('4H').apply(lambda x: (x.max() - x.min())).dropna()
df_m5['atr_h4'] = df_h4_atr.reindex(df_m5.index, method='ffill').shift(1)
df_m5['volatility_norm'] = df_m5['atr_h4'] / (df_m5['trend_d1'].rolling(100).mean() + 1e-6)

# Cleanup
df_m5.fillna(method='bfill', inplace=True)
df_m5.fillna(0, inplace=True)

# --------------------------------------------------------------------------------
# MARKET REGIME LABELING (REPLACING TRIPLE BARRIER)
# --------------------------------------------------------------------------------
print("🏷️  APLICANDO ETIQUETADO DE REGÍMENES (Rango/Alcista/Bajista)...")

conditions = [
    # RÉGIMEN 0: RANGO NEUTRAL (Baja Volatilidad + Precio Atrapado entre H1 y H4)
    (df_m5['volatility_norm'] < 0.005) & ((df_m5['pos_vs_h1'] - 1).abs() < 0.002),
    
    # RÉGIMEN 1: TENDENCIA ALCISTA (M5 > H1 > H4, y Tendencia H1 subiendo)
    (df_m5['pos_vs_h1'] > 1.001) & (df_m5['pos_vs_h4'] > 1.001) & (df_m5['trend_h1'] > df_m5['trend_h1'].shift(20)),
    
    # RÉGIMEN 2: TENDENCIA BAJISTA (M5 < H1 < H4, y Tendencia H1 bajando)
    (df_m5['pos_vs_h1'] < 0.999) & (df_m5['pos_vs_h4'] < 0.999) & (df_m5['trend_h1'] < df_m5['trend_h1'].shift(20))
]
choices = [0, 1, 2] # 0=Rango, 1=Alcista, 2=Bajista
df_m5['market_regime'] = np.select(conditions, choices, default=0) # Default to Range

# Distribution Check
counts = df_m5['market_regime'].value_counts()
print(f"📊 Regime Distribution:\n{counts}")

# 3. PREPARE INPUTS
# Features M5 Updated (9 Features: 3 Base + 3 Causal + 3 Context)
feature_cols = ['log_ret', 'rvol', 'volatility', 'entropy_z', 'fractal_higuchi', 'nofi', 'pos_vs_h1', 'pos_vs_h4', 'volatility_norm']
features_m5 = df_m5[feature_cols].values

# We keep D1 context separate for the bi-headed model? 
# DeepSeek suggests integrating all context. 
# Let's keep the Bi-Headed architecture but feed the new context features into the M5 head as well, 
# and keep D1 raw features for the second head as a "macro anchor".
feature_cols_d1 = ['trend_d1', 'volatility_norm', 'atr_h4'] # Simplified Macro features
# We need to construct these D1 features aligned with M5
features_d1 = df_m5[feature_cols_d1].values

labels = df_m5['market_regime'].values
y_categorical = tf.keras.utils.to_categorical(labels, num_classes=3)

# 4. SCALING & SEQUENCING
SEQ_LEN = 60
split_idx = int(len(features_m5) * 0.8)

# Winsorize & Scale M5 Features
df_train_m5 = pd.DataFrame(features_m5[:split_idx], columns=feature_cols)
df_val_m5 = pd.DataFrame(features_m5[split_idx:], columns=feature_cols)
df_train_m5_scaled, scaler_m5 = winsorize_and_scale(df_train_m5, feature_cols)
train_m5_norm = df_train_m5_scaled.values
val_m5_norm = scaler_m5.transform(df_val_m5)

# Scale D1 Features (Robust Scaler only)
scaler_d1 = RobustScaler()
train_d1_norm = scaler_d1.fit_transform(features_d1[:split_idx])
val_d1_norm = scaler_d1.transform(features_d1[split_idx:])

print("Tx Generating Dual Sequences...")
def create_dual_sequences(data_m5, data_d1, labels, seq_len):
    X_m5, X_d1, y = [], [], []
    for i in range(seq_len, len(data_m5)):
        X_m5.append(data_m5[i-seq_len:i])
        X_d1.append(data_d1[i]) # Static context for this timestep
        y.append(labels[i])
    return np.array(X_m5), np.array(X_d1), np.array(y)

X_train_m5, X_train_d1, y_train = create_dual_sequences(train_m5_norm, train_d1_norm, y_categorical[:split_idx], SEQ_LEN)
X_val_m5, X_val_d1, y_val = create_dual_sequences(val_m5_norm, val_d1_norm, y_categorical[split_idx:], SEQ_LEN)

# 5. MODEL ARCHITECTURE (UPDATED FOR 9 INPUTS)
def build_multitemporal_model(input_shape_m5, input_shape_d1):
    # --- RAMA 1: M5 Sequence (Now with H1/H4 Context embedded) ---
    in_m5 = Input(shape=input_shape_m5, name='Input_M5')
    
    # Spatial Dropout (Maintained)
    x = SpatialDropout1D(0.3)(in_m5)
    
    # LSTM Layers
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(x)
    x = Dropout(0.4)(x)
    x = LSTM(32)(x)
    x = BatchNormalization()(x)
    
    # --- RAMA 2: D1 Static Context ---
    in_d1 = Input(shape=input_shape_d1, name='Input_D1')
    y = Dense(16, activation='relu')(in_d1)
    y = BatchNormalization()(y)
    
    # --- FUSION ---
    concat = Concatenate()([x, y])
    z = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(concat)
    z = Dropout(0.5)(z)
    
    # Output: 3 Classes (Range, Bull, Bear)
    out = Dense(3, activation='softmax')(z)
    
    model = Model(inputs=[in_m5, in_d1], outputs=out)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # [DEEPSEEK ADVISED]: Use standard loss for initial Regime testing
        metrics=['accuracy']
    )
    return model

# 9 Features M5, 3 Features D1
model = build_multitemporal_model((SEQ_LEN, 9), (3,))
print("🤖 MODELO TITAN V5 (MULTITEMPORAL) COMPILADO.")

# 6. TRAINING
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUTPUT_PATH, 'titan_v5_deepseek.keras'),
    monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Class Weights (Regime Balancing)
# Since regimes might be imbalanced (more ranges than trends), calculate weights
y_integers = np.argmax(y_train, axis=1)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
cw_dict = dict(enumerate(cw))
print(f"⚖️ Class Weights: {cw_dict}")

print("🚀 INICIANDO ENTRENAMIENTO V5...")
history = model.fit(
    [X_train_m5, X_train_d1], y_train,
    epochs=50,
    batch_size=64,
    validation_data=([X_val_m5, X_val_d1], y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=cw_dict,
    verbose=1
)

joblib.dump(scaler_m5, os.path.join(OUTPUT_PATH, 'scaler_m5.pkl'))
joblib.dump(scaler_d1, os.path.join(OUTPUT_PATH, 'scaler_d1.pkl'))
print("🏆 OPERACIÓN V5 DEEPSEEK COMPLETADA.")
