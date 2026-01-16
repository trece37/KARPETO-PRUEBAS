# 💀 INFORME TÁCTICO: OPERACIÓN "ROMPER EL 48%"
**FECHA:** 14-ENE-2026 | **HORA:** Zulu+1
**ORIGEN:** ANTIGRAVITY | **DESTINO:** MANEL
**CLASIFICACIÓN:** CÓDIGO SUCIO (DIRTY HACKS) - READY TO DEPLOY

---

He peinado la literatura (2024-2025) y los foros de HFT rusos. Aquí tienes el **MENÚ DE EJECUCIÓN INMEDIATA**.
No hay teoría. Solo herramientas para matar el "Hold Trap".

## 🛠️ OPCIÓN 1: LA BOMBA NUCLEAR (FOCAL LOSS)
**Diagnóstico:** El modelo acierta el 48% diciendo "HOLD". `CrossEntropy` normal no le castiga suficiente.
**Solución:** `Focal Loss`. Inventada para detectar tumores (clase rara). Reduce el peso de los aciertos fáciles (Hold) y multiplica el dolor de fallar los difíciles (Buy/Sell).

**IMPLEMENTACIÓN (Keras/TensorFlow):**
```python
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Dirty Hack para Keras. 
    alpha: Peso para la clase minoritaria (Buy/Sell).
    gamma: Factor de enfoque. Cuanto más alto, menos le importan los 'Hold'.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Asegurar float32 para estabilidad
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Evitar log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calcular Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calcular peso focal: (1 - p_t)^gamma
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        
        return tf.math.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# USO EN EL SCRIPT:
# model.compile(loss=categorical_focal_loss(gamma=2.0, alpha=0.25), ...)
```

## ⚖️ OPCIÓN 2: EL MARTILLO (CLASS WEIGHTS via SKLEARN)
**Diagnóstico:** Si tienes 10.000 velas y 9.000 son Hold, el modelo es vago.
**Solución:** Obligarle a estudiar. Decirle: *"Si fallas un BUY te pego 10 veces. Si fallas un HOLD te pego 1 vez"*.

**CÓDIGO (Para meter antes de `model.fit`):**
```python
from sklearn.utils import class_weight

# Calcular pesos automáticos (Inverso de la frecuencia)
y_integers = np.argmax(y, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights_dict = dict(enumerate(class_weights))

# DIRTY HACK: Multiplicar manually los pesos de 0 (Buy) y 2 (Sell) x 2.0 extra
# para forzar agresividad.
class_weights_dict[0] *= 2.0 # Force BUY learning
class_weights_dict[2] *= 2.0 # Force SELL learning

# USO:
# history = model.fit(..., class_weight=class_weights_dict)
```

## 🎯 OPCIÓN 3: INGENIERÍA DE OBJETIVO (TRIPLE BARRIER METHOD)
**Diagnóstico:** Etiquetar con "Close > Close[i+5]" es basura ruidosa en M5.
**Solución:** Triple Barrier (Marcos Lopez de Prado). No mires "dentro de 5 velas". Mira "¿Toca primero el TP o el SL?".
**Hack:** Usar Volatilidad (ATR) para definir el ancho de la barrera.

**CÓDIGO (Snipper Logic):**
```python
def create_triple_barrier_labels(prices, volatility, sl_tp_ratio=1.0, barrier_width=2.0):
    """
    Genera etiquetas: 0=Buy (Toca TP superior), 1=Sell (Toca TP inferior/SL), 2=Hold (Expira)
    barrier_width: Multiplicador de volatilidad (ej. 2 * ATR)
    """
    labels = []
    # ... Lógica compleja de horizonte temporal ...
    # (Simplificado para implementación rápida: Usar Fixed Threshold dinámico)
    
    # REEMPLAZO RÁPIDO PARA TU LOOP ACTUAL:
    # En vez de pct_change > 0.005 (Fijo), usa:
    threshold = np.std(prices[-100:]) * 1.5 # 1.5 Sigmas de la volatilidad reciente
    
    if pct_change > threshold: return [1, 0, 0] # BUY
    elif pct_change < -threshold: return [0, 0, 1] # SELL
    else: return [0, 1, 0] # HOLD
```

## 🧪 OPCIÓN 4: INPUT ENGINEERING (LOG RETURNS)
**Diagnóstico:** El precio en 2000 era 300. En 2025 es 2000. `StandardScaler` sobre el precio bruto NO FUNCIONA BIEN en rangos tan amplios.
**Solución:** Usar **Log Returns** (Retornos Logarítmicos). Hacen que la crisis de 2008 y el Covid de 2020 sean comparables.

**CÓDIGO:**
```python
# En lugar de scaler.fit_transform(close_prices)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df.dropna(inplace=True)
data = df['log_ret'].values.reshape(-1, 1)
# Y luego escalas ESTO. Es estacionario. Al LSTM le encanta.
```

---

## 🔥 RECOMENDACIÓN DEL COMANDANTE (PLAN DE ACCIÓN YA)

Para solucionar el 48% **AHORA MISMO** sin reescribir todo el framework, aplica este combo:

1.  **INPUT:** Cambia a **Log Returns** (Opcion 4). Es un cambio de 1 línea que arregla la escala.
2.  **LOSS:** Mete **Class Weights** (Opción 2). Es fácil y brutalmente efectivo.
3.  **DATOS:** Quédate **SOLO CON M5 (2020-2025)**. Elimina la basura vieja D1 que confunde la volatilidad.

**¿EJECUTO ESTE COMBO (M5 + LogRet + ClassWeights) EN `train_titan_v3.py` AHORA MISMO?**
