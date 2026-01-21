# 📖 ENCICLOPEDIA DEL BOT (ÍNDICE MAESTRO)
**Autor:** Antigravity (Tu Nena) | **Para:** Manel (Arquitecto)
**Protocolo:** 666RULES (Kién, Ké, Kómo) | **Versión:** 1.0

> **⚠️ BARRIGA RULE:** Si no entiendes el índice, no entenderás el mapa. Este documento es la brújula conceptual de AKILES1337.

---

## 🏛️ TABLA DE CONTENIDOS (EL MAPA DEL TESORO)

1.  **[GLOSARIO CONCEPTUAL](#1-glosario-conceptual-el-lenguaje-del-imperio)**
2.  **[FASE 1: INFRAESTRUCTURA (EL PUENTE ZMQ)](#2-fase-1-infraestructura-el-puente-zmq)**
3.  **[FASE 2: CEREBRO (INTELIGENCIA PURA)](#3-fase-2-cerebro-inteligencia-pura)**
4.  **[FASE 3: MODELOS & ESTRATEGIA (EL ALMA)](#4-fase-3-modelos--estrategia-el-alma)**
5.  **[FASE 4: FEATURES & ENTRENAMIENTO (EL GIMNASIO)](#5-fase-4-features--entrenamiento-el-gimnasio)**
6.  **[FASE 5: EL OBRERO (EJECUCIÓN)](#6-fase-5-el-obrero-ejecución)**

---

## 1. GLOSARIO CONCEPTUAL (EL LENGUAJE DEL IMPERIO)

Antes de tocar código, definimos la **Verdad**.

### 🔹 ZMQ Bridge (El Teléfono Rojo)
*   **KIÉN:** ZeroMQ (ZMQ). Una librería de mensajería ultrarrápida.
*   **KÉ:** Un "puente de fibra óptica" entre Python (Cerebro) y MT5 (Músculo).
*   **KÓMO:** Sustituye archivos de texto lentos (`orden.txt`) por **Sockets** en memoria RAM. Permite que el bot reaccione en microsegundos, no en segundos. Es la diferencia entre un Walkie-Talkie y Telepatía.

### 🔹 Seldon (El Canario en la Mina)
*   **KIÉN:** Algoritmo de Detección de Anomalías (`EllipticEnvelope`).
*   **KÉ:** Un sistema de defensa pasiva. No predice precios, predice **PELIGRO**.
*   **KÓMO:** Aprende lo que es "normal" en el mercado. Si ve algo raro (Crash, Flash Crash, Manipulación), grita **VETO** y paraliza todas las operaciones. Es tu seguro de vida.

### 🔹 Titan V3 (El Cerebro)
*   **KIÉN:** Tu motor principal en Python (`src/brain`).
*   **KÉ:** Donde vive la lógica, los modelos IA y la gestión de riesgo.
*   **KÓMO:** Orquesta todo. Recibe datos, consulta a Seldon, consulta a LSTM, decide el riesgo, y envía la orden final al Obrero.

---

## 2. FASE 1: INFRAESTRUCTURA (EL PUENTE ZMQ)
*Estado: Pendiente de Ejecución*

Esta fase construye las carreteras antes de fabricar los coches.
*   📁 **`mql-zmq-master`**: La caja de herramientas. Contiene las DLLs y archivos `.mqh` para enseñar a MetaTrader a hablar ZMQ.
*   **Objetivo:** Instalar el servidor en MT5 y el cliente en Python. Verificar que se saludan ("Ping-Pong").

## 3. FASE 2: CEREBRO (INTELIGENCIA PURA)
*Estado: En Diseño*

Aquí definimos cómo piensa el bot.
*   **Arquitectura Alpha:** Separar la predicción (`AlphaModel`) de la ejecución (`Portfolio`).
*   **Persistencia:** Que el bot recuerde lo que hizo ayer (Base de datos / Estado).

## 4. FASE 3: MODELOS & ESTRATEGIA (EL ALMA)
*Estado: En Diseño*

Los componentes matemáticos.
*   **LSTM:** La red neuronal que predice la dirección.
*   **Risk Manager:** Las reglas de cuánto apostar (Position Sizing).

## 5. FASE 4: FEATURES & ENTRENAMIENTO (EL GIMNASIO)
*Estado: En proceso (01_LAB)*

La preparación de los atletas.
*   **WFO (Walk-Forward Optimization):** Entrenar como si viviéramos el pasado, sin mirar el futuro.
*   **Feature Engineering:** Crear los indicadores que alimentan a la IA.

## 6. FASE 5: EL OBRERO (EJECUCIÓN)
*Estado: Existente (V1/V2)*

El soldado final.
*   **Expert Advisor (.mq5):** El script que vive en MetaTrader. Ya no piensa, solo obedece órdenes ciegas que llegan por el ZMQ Bridge.

---
*Fin del Índice Maestro. Máximo 1000 palabras.*
