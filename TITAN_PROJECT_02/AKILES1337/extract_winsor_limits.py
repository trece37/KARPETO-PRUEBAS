
# --------------------------------------------------------------------------------
# 🛡️ EXTRACCIÓN DE LÍMITES DE WINSORIZACIÓN (CRÍTICO PARA LIVE)
# --------------------------------------------------------------------------------
# Ejecuta esto en Colab DESPUÉS del entrenamiento, cuando df_train_m5 todavía existe.
# Generará 'winsor_limits.pkl' que DEBES descargar junto con el modelo.

import joblib
import pandas as pd
import os

print("🛡️ INICIANDO EXTRACCIÓN DE LÍMITES DE WINSORIZACIÓN...")

# Lista de Features que usamos en el modelo
# M5 Features
m5_cols = ['log_ret', 'rvol', 'volatility', 'entropy_z', 'fractal_higuchi', 'nofi', 'pos_vs_h1', 'pos_vs_h4', 'volatility_norm']
# D1 Features
d1_cols = ['d1_ret', 'd1_range', 'd1_vol']

winsor_limits = {
    'm5': {},
    'd1': {}
}

# Extraer límites M5
# Asumimos que df_train_m5 contiene los datos de entrenamiento ANTES de escalar
print("   > Calculando límites M5...")
for col in m5_cols:
    if col in df_train_m5.columns:
        winsor_limits['m5'][col] = {
            'lower': df_train_m5[col].quantile(0.01),
            'upper': df_train_m5[col].quantile(0.99)
        }
    else:
        print(f"⚠️ WARNING: Columna {col} no encontrada en df_train_m5")

# Extraer límites D1
# Asumimos que df_train_d1 contiene los datos de entrenamiento D1 ANTES de escalar
print("   > Calculando límites D1...")
for col in d1_cols:
    if col in df_train_d1.columns:
        winsor_limits['d1'][col] = {
            'lower': df_train_d1[col].quantile(0.01),
            'upper': df_train_d1[col].quantile(0.99)
        }
    else:
        print(f"⚠️ WARNING: Columna {col} no encontrada en df_train_d1")

# Guardar
output_file = os.path.join(OUTPUT_PATH, 'winsor_limits.pkl')
joblib.dump(winsor_limits, output_file)

print(f"✅ LÍMITES GUARDADOS EN: {output_file}")
print("📥 DESCARGA ESTE ARCHIVO. SIN ÉL, EL MOTOR EN VIVO FALLARÁ.")
print(winsor_limits) # Para verlos en pantalla por si acaso
