# 🏗️ ESTRUCTURA DE TRABAJO: PROTOCOLO DE ANÁLISIS (EL TAMIZ)
**FECHA:** 14-ENE-2026
**OBJETIVO:** FILTRAR LA ENCICLOPEDIA DE SOLUCIONES Y SELECCIONAR LOS "DIRTY HACKS" GANADORES.

---

## 1. PRINCIPIOS DE SELECCIÓN (EL FILTRO RICK)
No voy a probar todo. Solo lo que pase este filtro de 3 capas:

*   **CAPA 1: INMEDIATEZ (Velocidad de Despliegue)**
    *   ¿Se puede implementar en `train_titan_v3.py` en menos de 15 minutos?
    *   SI: Pasa.
    *   NO (Requiere reescribir todo el bot, cambiar arquitectura a Transformers desde cero): A LA BASURA.

*   **CAPA 2: PUREZA DE CÓDIGO (Dependencias)**
    *   ¿Usa librerías estándar (`tensorflow`, `numpy`, `pandas`, `sklearn`)?
    *   SI: Pasa.
    *   NO (Requiere instalar librerías exóticas, compilar C++, o APIs externas): A LA BASURA.

*   **CAPA 3: PUNTERÍA (Target: Hold Trap)**
    *   ¿Ataca específicamente el problema del `val_accuracy: 0.48` (Class Imbalance)?
    *   SI (Loss functions, Sampling, Weights): Pasa con PRIORIDAD ALTA.
    *   NO (Mejoras genéricas de optimización, visualización): PRIORIDAD BAJA.

---

## 2. METODOLOGÍA DE PROCESAMIENTO (LA MÁQUINA)

Una vez reciba tus documentos en `TEMPORAL`, ejecutaré este bucle:

1.  **LECTURA RAPIDA:** Escaneo de palabras clave (`Focal Loss`, `ADF`, `Wavelet`, `GARCH`, `Attention`).
2.  **EXTRACCIÓN DE CÓDIGO:** Copiaré solo los bloques de código Python. Ignoraré la literatura.
3.  **RANKING DE VIABILIDAD:** Clasificaré cada solución en:
    *   🔴 **DESCARTADO:** Demasiado complejo / Teórico.
    *   🟡 **RESERVA:** Interesante pero lento.
    *   🟢 **APLICAR YA:** Código listo para copiar/pegar en nuestro script.

---

## 3. RESULTADO FINAL (LO QUE TE ENTREGARÉ)
Generaré un archivo único: **`PLAN_DE_BATALLA_FINAL.md`**.

Contendrá:
1.  **El "Menú Degustación":** Las 3 mejores soluciones seleccionadas de tu Enciclopedia.
2.  **El Script Final (`train_titan_v3_ULTIMATE.py`):** Con la solución ganadora ya integrada.

**ESTOY LISTO PARA RECIBIR LA ENCICLOPEDIA. SÚBELA A `00_FACTORY/TEMPORAL`.**
