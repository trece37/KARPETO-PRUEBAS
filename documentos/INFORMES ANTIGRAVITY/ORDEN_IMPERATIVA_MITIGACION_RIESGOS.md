# ⚔️ ORDEN IMPERATIVA: PROTOCOLO DE MITIGACIÓN "ESCUDO DE HIERRO"
**PRIORIDAD:** DEFCON 1 | **EMISOR:** ANTIGRAVITY (MoE TRADER MODE)
**DESTINATARIO:** COMANDANTE MANEL
**OBJETIVO:** PARCHEAR LAS 5 GRIETAS DEL "PLATINUM+ COMBO" PARA EVITAR LA AUTODESTRUCCIÓN.

---

⚠️ **SITUACIÓN:** Tenemos un Fórmula 1. Corre mucho, pero es de cristal.
**ORDEN:** Ejecuta estos 5 PROTOCOLOS DE DEFENSA inmediatamente para blindar el sistema.

---

## 🛡️ 1. CONTRA LA CEGUERA MACRO (D1 Missing)
**👉 PROTOCOLO: EL PERISCOPIO HUMANO**
El modelo M5 es ciego a la pared que tiene delante. TÚ eres sus ojos.
**ACCIÓN INMEDIATA:**
*   **Regla de Oro:** Antes de encender el bot cada mañana, mira el gráfico D1.
*   **La Veto Rule:** Si D1 está tocando una Media Móvil de 200 o un Soporte/Resistencia histórico -> **APAGA EL BOT (O FUERZA "SOLO VENTAS/COMPRAS").**
*   **No dejes al niño conducir solo en la autopista.** Tú eres el copiloto estructural.

## 🛡️ 2. CONTRA LA FRAGILIDAD DEL "1.5" (Triple Barrier)
**👉 PROTOCOLO: RECALIBRACIÓN "WFO-LIGHT"**
El mercado cambia. El 1.5 de hoy es la basura de mañana.
**ACCIÓN INMEDIATA:**
*   **Ritmo:** Cada VIERNES a cierre de mercado.
*   **Tarea:** Ejecuta `train_titan_v3_ULTIMATE.py` con los datos de la última semana añadidos.
*   **Chequeo:** Si la accuracy baja del 50%, cambia el `barrier_width` a `1.0` o `2.0` y re-entrena una época rápida. Encuentra el nuevo multiplicador de la semana.
*   **Adaptarse o morir.**

## 🛡️ 3. CONTRA LA DEPENDENCIA DE VOLUMEN (Broker Slave)
**👉 PROTOCOLO: FIDELIDAD ABSOLUTA (DATA SOURCE LOCK)**
El `tick_volume` es el ADN de tu broker. Si cambias de broker, el ADN no coincide.
**ACCIÓN INMEDIATA:**
*   **Prohibido:** Cambiar de Broker o de Tipo de Cuenta (Standard a ECN, etc.) sin re-entrenar desde cero.
*   **Vigilancia:** Si tu broker anuncia "cambios en los servidores" o "nuevas horas de liquidez", asume que el modelo ha muerto. Re-entrena con los datos nuevos en cuanto tengas 2 semanas de historia.

## 🛡️ 4. CONTRA EL ERROR DE SCALER (Cisne Negro)
**👉 PROTOCOLO: ABRAZADERA DE SEGURIDAD (INPUT CLAMPING)**
Si el precio se va a la luna, el Scaler fallará.
**ACCIÓN INMEDIATA (MENTAL):**
*   Sabes que el Scaler se entrenó con precios de 2020-2025.
*   **La Alarma:** Si XAUUSD rompe máximos históricos con violencia (+5% en un día), **APAGA EL BOT.**
*   El modelo no sabe operar en "Terra Incognita". Espera a que se formen nuevos datos y re-entrena (Protocolo 2).

## 🛡️ 5. CONTRA EL SILENCIO DE RADIO (Eternal Hold)
**👉 PROTOCOLO: EL DESFIBRILADOR**
Un modelo que no opera no pierde dinero, pero tampoco lo gana.
**ACCIÓN INMEDIATA:**
*   **Monitor de Pulso:** Si el bot pasa **48 HORAS** sin abrir una sola operación...
*   **Diagnóstico:** El mercado está en calma chicha y la Barrera 1.5 es demasiado ancha.
*   **Inyección de Adrenalina:** Baja manual y temporalmente la barrera en el código (ej. a `1.0`) y re-carga el modelo. Oblígalo a ser más sensible.

---

## 📜 RESUMEN DE LA ORDEN
1.  **MIRA D1 TÚ MISMO.**
2.  **RE-ENTRENA LOS VIERNES.**
3.  **NO CAMBIES DE BROKER.**
4.  **APAGA EN CRASH/BOOM HISTÓRICO.**
5.  **SI SE CALLA 2 DÍAS, BAJA LA BARRERA.**

**CUMPLE ESTA ORDEN Y EL FÓRMULA 1 LLEGARÁ A META.**
**FALLA EN ESTO Y TE ESTRELLARÁS EN LA PRIMERA CURVA.**

**¡EJECUCIÓN!** 🔥
