# 🦅 MATRIZ DE SOLUCIONES CRÍTICAS (FORENSE MULTI-IA)
**FECHA:** 14-ENE-2026 | ** FUENTE:** CHATGPT + CLAUDE + GROK + QWEN
**OBJETIVO:** REPARAR LAS 5 GRIETAS DEL "FORMULA 1" (PLATINUM+ ARCHITECTURE)

---

## 🏆 EL VEREDICTO DE LA COLMENA (SIÍNTESIS TÉCNICA)

He cruzado las respuestas de las 4 Inteligencias Artificiales de élite. Todas coinciden en los puntos de fallo y ofrecen soluciones muy similares.
Aquí está la **MATRIZ DE DECISIÓN** para aplicar los parches definitivos.

| PROBLEMA | 🥇 GANADOR (LA SOLUCIÓN ELEGIDA) | 🥈 FINALISTA | ¿POR QUÉ GANÓ? |
| :--- | :--- | :--- | :--- |
| **1. CEGUERA MACRO** | **DUAL-STREAM LSTM (BICÉFALO)** | Embedding D1 Estático | **Claude & Qwen:** Separar el flujo rápido (M5) del lento (D1) es lo más robusto. No ensucia la serie M5 con datos repetidos. |
| **2. BARRERA FRÁGIL** | **VOLATILITY SCALING** ($k_t \cdot \frac{Vol_{now}}{Vol_{avg}}$) | Quantile Barriers | **ChatGPT & Grok:** Ajustar el ancho de la barrera proporcionalmente al régimen de volatilidad actual es simple, matemático y efectivo. |
| **3. VOLUMEN SLAVE** | **RVOL (TIME-OF-DAY)** | Rank Transform | **TODOS:** El volumen relativo (Volumen Actual / Promedio Histórico para esa Hora) elimina el sesgo del broker y la estacionalidad del día. |
| **4. SCALER ROTO** | **ROBUST SCALER (QUANTILES)** | Windowed Log-Returns | **Standard:** Usar percentiles (25-75 o 5-95) ignora los outliers extremos (ATH violentos) sin romper la escala de datos normales. |
| **5. MODELO MUDO** | **BIAS INJECTION + DYNAMIC THRESHOLD** | Isotonic Calibration | **Dirty Hack Puro:** Forzar matemáticamente a la última capa a "querer" operar más (bias > 0) y bajar el umbral de decisión si la volatilidad es baja. |

---

## 🛠️ EL "KIT DE REPARACIÓN" (LO QUE VAMOS A CODIFICAR)

Para solucionar esto **SIN REHACER TODO DESDE CERO**, propongo este **Plan Quirúrgico**:

### 1. 🔭 PARCHE BICÉFALO (SOLUCIÓN MACRO)
*   **Acción:** Modificar `build_ultimate_model` para aceptar **DOS INPUTS**: `(60, 3)` para M5 y `(5, 4)` para D1 (últimos 5 días: Close, High, Low, Vol).
*   **Código:** Crear una rama paralela pequeña para D1 y concatenarla antes del Dense final.

### 2. 🌊 PARCHE DE FLUIDOS (SOLUCIÓN VOLUMEN)
*   **Acción:** Reemplazar `log_vol` con **`RVOL`**.
*   **Lógica:** Calcular la media de volumen *para cada hora del día* en el train set, y dividir el volumen actual por esa media.

### 3. 🛡️ PARCHE ADAPTATIVO (SOLUCIÓN BARRERA)
*   **Acción:** Cambiar la función `apply_triple_barrier`.
*   **Lógica:** Calcular `volatility_ratio = current_vol / rolling_mean_vol`. Multiplicar el ancho `1.5` por este ratio. Si el mercado está loco, la barrera se aleja. Si está muerto, se acerca.

### 4. 🎚️ PARCHE DE VALENTÍA (SOLUCIÓN SILENCIO)
*   **Acción:** En la capa de salida `Dense(3, activation='softmax')`, inyectar `bias_initializer`.
*   **Código:** `bias_initializer=tf.keras.initializers.Constant([-1.0, 0.5, 0.5])` (Castigar Hold, premiar Buy/Sell desde el nacimiento).

---

## ⚠️ ¿CONFIRMAS EL DESPLIEGUE?
Manel, si das luz verde, reescribo `train_titan_v3_ULTIMATE.py` **AHORA MISMO** aplicando estos 4 parches.
El modelo pasará de ser un **Fórmula 1 de Cristal** a un **Tanque de Carreras**.

**¿PROCEDEMOS A LA CIRUGÍA?** 🔪🔥
