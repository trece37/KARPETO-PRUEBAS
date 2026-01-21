# --------------------------------------------------------------------------------
# 🏋️ ARCHIVO: SRC/BRAIN/TRAINING/TRAINER_GPU.PY
# PROYECTO: ACHILLES TITAN V3
# AUTOR: ANTIGRAVITY (Ejecutando Instrucciones de Manel)
# FUENTE: Dirty 100 (Sección D: Execution)
# PROCESO: WFO / TRAIN / VALIDATE
# --------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings

# Imports Locales (Estructura Titan)
from config.settings import TradingConfig
from src.brain.features.feature_engineering import FeatureEngineer
from src.brain.models.lstm import AchillesLSTM

warnings.filterwarnings("ignore")

class TitanTrainer:
    """
    ORQUESTADOR DE ENTRENAMIENTO GPU.
    Conecta: Datos Crudos -> FeatureEngineer -> Tensores -> AchillesLSTM
    """
    def __init__(self):
        self.cfg = TradingConfig
        self.fe = FeatureEngineer()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_prep_data(self, csv_path):
        """ Carga y prepara los datos usando el FeatureEngineer """
        print(f"💀 [TRAINER] Cargando datos de: {csv_path}")
        
        # 1. Carga
        try:
            df = pd.read_csv(csv_path)
            # Normalización de nombres de columnas (strip)
            df.columns = df.columns.str.strip()
            
            # Conversión de Fecha
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
                
            print(f"   ✅ Datos Crudos: {len(df)} filas.")
        except Exception as e:
            raise ValueError(f"❌ Error cargando CSV: {e}")

        # 2. Ingeniería de Features (Dirty Hacks)
        df_processed = self.fe.process_all(df)
        
        # 3. Etiquetado (Labelling) para Clasificación (Hold/Buy/Sell)
        # Lógica simple para V3:
        # Buy (1) si Close futuro (t+1) > Close actual + umbral
        # Sell (2) si Close futuro (t+1) < Close actual - umbral
        # Hold (0) caso contrario
        
        # Definir un umbral mínimo de movimiento para considerar señal (evitar ruido)
        # Usamos ATR si existe, sino un % fijo.
        threshold = 0.0 # Por defecto, cualquier movimiento cuenta (Dirty Clasico)
        if 'ATR' in df_processed.columns:
            threshold = df_processed['ATR'] * 0.1 # 10% del ATR como filtro de ruido
        
        future_close = df_processed['Close'].shift(-self.cfg.FORECAST_HORIZON)
        change = future_close - df_processed['Close']
        
        conditions = [
            (change > threshold), # Buy
            (change < -threshold) # Sell
        ]
        choices = [1, 2] # 1=Buy, 2=Sell, default=0 (Hold)
        
        df_processed['Target'] = np.select(conditions, choices, default=0)
        
        # Eliminar filas con NaN generados por el shift del Target
        df_processed.dropna(inplace=True)
        
        print(f"   ✅ Datos Etiquetados. Distribución de Clases:\n{df_processed['Target'].value_counts()}")
        
        return df_processed

    def create_tensors(self, df):
        """ Convierte DataFrame a Tensores 3D (X) y One-Hot (y) """
        print("💀 [TRAINER] Tensorizando datos...")
        
        # Separar Features de Target
        # Excluir columnas no numéricas o target
        target_col = 'Target'
        drop_cols = [target_col, 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'] # Time features no se escalan igual a veces, pero aqui las incluimos en X
        # Re-incluir time features en X pero cuidado con scaler. Escalar todo es la norma Dirty.
        feature_cols = [c for c in df.columns if c != target_col]
        
        data_x = df[feature_cols].values
        data_y = df[target_col].values
        
        # Escalado (Fit sobre todo el dataset por simplificacion en Fase 4 inicial, 
        # en Fase 5 WFO se hace fit solo en train)
        x_scaled = self.scaler.fit_transform(data_x)
        
        # Guardar Scaler
        os.makedirs(self.cfg.MODELS_PATH, exist_ok=True) # Asegurar directorio
        joblib.dump(self.scaler, os.path.join(self.cfg.MODELS_PATH, "titan_scaler.pkl"))
        
        X, y = [], []
        window = self.cfg.LOOKBACK_WINDOW
        
        for i in range(window, len(x_scaled)):
            X.append(x_scaled[i-window:i])
            y.append(data_y[i]) # Target en t (que ya mira al futuro t+1 por el shift previo)
            
        X = np.array(X)
        y = to_categorical(y, num_classes=3) # [0, 0, 1]
        
        print(f"   ✅ Tensores Listos: X={X.shape}, y={y.shape}")
        return X, y

    def run_training(self, data_path=None):
        """ Ejecuta el Pipeline Completo """
        print("⚫⚫⚫ INICIANDO ENTRENAMIENTO TITAN V3 (GPU) ⚫⚫⚫")
        
        # Ruta de datos: Si no se pasa, usa la de Config
        csv_source = data_path if data_path else self.cfg.DATA_RAW
        if not os.path.exists(csv_source):
             # Fallback a un archivo de ejemplo si no existe el raw
             print(f"⚠️ NO DATA FOUND AT {csv_source}. Buscando en directorio data/...")
             potential_files = [f for f in os.listdir(self.cfg.DATA_PATH) if f.endswith('.csv')]
             if potential_files:
                 csv_source = os.path.join(self.cfg.DATA_PATH, potential_files[0])
             else:
                 raise FileNotFoundError("❌ CRITICAL: No CSV files found to train.")

        # 1. Pipeline de Datos
        df = self.load_and_prep_data(csv_source)
        X, y = self.create_tensors(df)
        
        # 2. Split (80/20 Hold-out simple para esta fase)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 3. Instanciar Modelo
        # Input shape: (Timesteps, Features)
        input_shape = (X_train.shape[1], X_train.shape[2])
        titan = AchillesLSTM(input_shape=input_shape)
        model = titan.model
        
        print(f"💀 [MODELO] Compilado. Arquitectura:\n")
        model.summary()
        
        # 4. Callbacks (Dirty 100 Standards)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            ModelCheckpoint(
                filepath=os.path.join(self.cfg.MODELS_PATH, "titan_v3_best.keras"),
                monitor='val_accuracy', # O val_loss
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 5. Fit
        print("💀 [GPU] Disparando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.EPOCHS,
            batch_size=self.cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ ENTRENAMIENTO FINALIZADO. Modelo guardado en models/.")
        return history

if __name__ == "__main__":
    trainer = TitanTrainer()
    trainer.run_training()
