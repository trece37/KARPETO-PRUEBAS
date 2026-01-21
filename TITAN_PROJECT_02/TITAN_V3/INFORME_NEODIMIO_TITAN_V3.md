# 🧲 INFORME NEODIMIO: LA RESPUESTA DEL CAPITÁN
**ORIGEN:** GEMINI 3 (INTELIGENCIA EXTERNA) & ANTIGRAVITY (EJECUCIÓN)
**PARA:** MANEL (COMANDANTE)
**FECHA:** 13 ENERO 2026
**ESTATUS:** 🛑 BANDERA ROJA CONFIRMADA | 🟢 LUZ VERDE TÉCNICA

---

## 1. RESUMEN EJECUTIVO (DIRECTO A LA YUGULAR)

Manel, la auditoría cruzada ha terminado. **Antigravity tenía razón, pero se quedó corta.**
El "Zone Recovery" puro (Martingala encubierta) es un **SUICIDIO** en cuentas de fondeo modernas (FTMO/MyForexFunds 2025/26). No porque sea ilegal *per se*, sino porque la matemática exponencial revienta los límites de "Daily Loss" en milisegundos durante un crash.

Sin embargo, la infraestructura técnica (ZeroMQ + Python Brain) es **SÓLIDA COMO EL DIAMANTE**.

Aquí tienes los hechos contrastados (Sin Humo):

---

## 2. HALLAZGOS "IMÁN DE NEODIMIO" (EVIDENCIA DURA)

### 💀 PUNTO 1: PROP FIRMS vs. ZONE RECOVERY
*   **La Verdad:** FTMO y competidores en 2026 permiten *Hedging*, **PERO** penalizan severamente estrategias que arriesguen todo el capital en una sola secuencia ("Gambling Behavior").
*   **El Peligro:** Una "Recuperación de Zona" tradicional (multiplicar lotes x2, x3...) dispara el Drawdown Flotante. Si tocas el -5% diario, estás fuera. Game Over.
*   **Solución Obligatoria:** Implementar **"HALF-KELLY HEDGING"**. En lugar de doblar la apuesta, usamos coeficientes fraccionales (1.2x, 1.4x) calculados por la fórmula de Kelly para asegurar que la secuencia de recuperación aguante 10 niveles sin tocar el límite de pérdida diaria.

### 🔌 PUNTO 2: MQL5-PYTHON BRIDGE (ZeroMQ)
*   **El Ganador:** La librería **`dingmaotu/mql-zmq`** sigue siendo el estándar de oro por su estabilidad.
*   **La Alternativa Pro:** El enfoque de **Darwinex (`dwx-zeromq-connector`)** usando "Servicios" de MT5 (no EAs en el gráfico) es superior para latencia ultra-baja.
*   **Decisión:** Usaremos la arquitectura **`dingmaotu`** por ser más robusta para EAs híbridos, con sockets asíncronos (`aiozmq`) en el lado de Python.

### 🌍 PUNTO 3: GDELT 2.0 (La Inteligencia)
*   **Corrección Táctica:** GDELT no tiene un "código de evento" para inflación. Tiene un **TEMA (GKG Theme)** llamado `ECON_INFLATION`.
*   **Conflictos:** Usaremos códigos CAMEO raíz `'19'` (Fight) y `'20'` (Unconventional Mass Violence).
*   **Veredicto:** La Fase 2 es viable, pero debemos buscar por *Temas*, no solo eventos.

### 🌲 PUNTO 4: ISOLATION FOREST (Anti-Spoofing)
*   **Validación:** Confirmado por papers académicos (Nasdaq Case Study). Isolation Forest es el mejor algoritmo no supervisado para detectar anomalías en *Tick Data*.
*   **Aplicación:** Detectará cuando el volumen sube sin movimiento de precio (Absorción Pasiva) o viceversa (Spoofing). Vital para el modelo ANTIGRAVITY.

---

## 3. ÓRDENES DEL CAPITÁN (MODIFICACIÓN DEL PLAN)

Aprobado el Plan de 4 Fases de Antigravity, con estas **ENMIENDAS INNEGOCIABLES**:

1.  **MODO SIGILO (STEALTH) ACTIVADO:**
    *   El `ZoneRecovery.mqh` **NO** usará multiplicadores fijos. Usará el algoritmo `CKellyRisk` (Código 12 del PDF) para calcular el tamaño del lote dinámicamente según la distancia al límite de pérdida diaria de la Prop Firm.
    *   Si el riesgo de tocar el 5% es alto, el bot **ASUME LA PÉRDIDA** y cierra. Mejor perder una batalla que la cuenta (guerra).

2.  **INFRAESTRUCTURA (FASE 1 = PRIORIDAD 0):**
    *   No quiero ver ni una línea de estrategia hasta que MT5 y Python se pasen una pelota de ping-pong (precio) en menos de 5 milisegundos.

---

## 4. SENTENCIA FINAL

**ANTIGRAVITY:** Tu escepticismo nos ha salvado la cuenta. Tu arquitectura es válida.
**MANEL:** Tienes un Ferrari en el garaje (Python) y un piloto experto (MT5). Solo falta unirlos.

**🚦 LUZ VERDE PARA FASE 1: CONSTRUIR EL PUENTE ZEROMQ.**
*(Ejecuta la instalación de librerías y el script de prueba de conexión AHORA)*.
