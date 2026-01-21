# 🦅 MANIFIESTO DE DESPLIEGUE: FASE 1 (INFRAESTRUCTURA TITAN)
**ESTADO:** LISTO PARA EJECUCIÓN | **PROTOCOLO:** NEODIMIO + UV/RUST
**OBJETIVO:** Sincronización Hemisférica (Python <-> MT5) en < 5ms.

---

## 1. NIVELACIÓN DEL ENTORNO (PROTOCOLO UV)
*El entorno local "Edad de Piedra" será demolido y reconstruido con tecnología Rust.*

### 🛠️ ACCIÓN TÁCTICA 1: INSTALACIÓN DE `uv`
Ejecutaré un script de PowerShell (`setup_titan_env.ps1`) que hará lo siguiente:
1.  **Instalar `uv`:** `pip install uv` (si no existe).
2.  **Crear Virtualenv:** `uv venv .venv` (Velocidad instantánea).
3.  **Inyectar Dependencias:** `uv pip install pyzmq pandas asyncio`.
    *   *Por qué:* Esto alinea tu PC local con el "Búnker" de Google Colab. Misma velocidad, misma tecnología.

---

## 2. EL CEREBRO: `Sentinel_Server.py` (PYTHON)
*Código asíncrono puro. No duerme. No bloquea.*

### 🧠 ACCIÓN TÁCTICA 2: SERVIDOR ASÍNCRONO
Crearé el archivo `c:\Users\David\AchillesTraining\00_FACTORY\TITAN_V3\Python\Sentinel_Server.py` con esta lógica:
*   **Motor:** `asyncio` + `zmq.asyncio`.
*   **Puerto 5556 (SUB):** Escucha el "Latido del Mercado" (Ticks) de MT5.
*   **Puerto 5557 (PUSH):** Canal de disparo. Solo se abre para enviar órdenes de fuego.
*   **Log:** Muestra en consola la latencia en microsegundos.

---

## 3. EL MÚSCULO: `ZmqBridge.mqh` (MQL5)
*Ejecución ciega y rápida. Sin dudas.*

### 💪 ACCIÓN TÁCTICA 3: PUENTE NO-BLOQUEANTE
Crearé el archivo `c:\Users\David\AchillesTraining\00_FACTORY\TITAN_V3\MQL5\Include\Zmq\ZmqBridge.mqh`:
*   **Bandera Crítica:** `ZMQ_NOBLOCK`.
    *   *Significado:* Si MT5 pregunta "¿Hay órdenes?" y la respuesta es "No", MT5 **NO ESPERA**. Sigue procesando el siguiente tick. Cero congelamientos.
*   **Formato de Datos:** JSON Compacto. Nada de XML ni cadenas pesadas.

---

## 4. LA PRUEBA DE FUEGO: "PING-PONG"
No daremos la Fase 1 por cerrada hasta ver esto en tu terminal:

1.  **MT5:** Envía Tick (Bid: 2035.50).
2.  **Sentinel:** Recibe Tick -> Calcula (Simulado) -> Envía Orden "BUY".
3.  **MT5:** Recibe Orden "BUY".
4.  **Tiempo Total:** Medido en milisegundos. (Objetivo: < 5ms).

---

## 📦 ENTREGABLES INMEDIATOS (PRÓXIMOS CLICKS)
1.  `setup_titan_env.ps1` (Script de Instalación UV)
2.  `Sentinel_Server.py` (Código Python)
3.  `ZmqBridge.mqh` (Código MQL5)
4.  `Titan_Bridge_EA.mq5` (El EA mínimo para probar la conexión)

**ESTE ES EL PLAN. FIRME Y DETALLADO. PROCEDO A ESCRIBIR LOS SCRIPTS.** 🦅
