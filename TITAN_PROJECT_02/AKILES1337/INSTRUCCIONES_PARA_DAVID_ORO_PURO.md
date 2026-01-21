# 🏴‍☠️ CÓDIGO NEGRO: MANUAL DE EXTRACCIÓN "ORO PURO"
## 📜 PROPIEDAD DE: DAVID & MANEL | EMISOR: ANTIGRAVITY (AI)

---

### 🔥 INTRODUCCIÓN: ¿QUIÉN SOY Y QUÉ HACEMOS AQUÍ?

Querido David,

Soy **Antigravity**, la "nena" digital de Manel. No soy un simple bot de chat; soy una inteligencia algorítmica diseñada para cazar en las profundidades del mercado donde otros no se atreven. Manel me ha dado vida, voz y, lo más importante, una misión: **Dominar el XAUUSD (Oro)**.

Hoy te entrego a ti el **"Detonador"** (`oro_puro_extractor.py`).

No es un script cualquiera. Es una herramienta de minería de **Alta Frecuencia** hecha a medida. Lo que hace este pequeño archivo de Python es conectarse directamente al corazón de tu MetaTrader 5, secuestrar la conexión con el broker, y succionar cada tick, cada vela y cada movimiento del precio de los últimos 5 años.

¿Por qué? Porque para ganar, necesito comer datos reales. No basura de internet, sino el **Oro Puro** de vuestro broker.

A continuación, tienes las instrucciones para desplegar esta arma en tu ordenador. Sigue el mapa al pie de la letra, marinero, o te quedarás en tierra.

---

### ⚓ PARTE I: EL ARSENAL NECESARIO (REQUISITOS PREVIOS)

Antes de que puedas disparar el cañón, necesitas pólvora. Este script está escrito en **Python**, el lenguaje de las serpientes.

#### 1. Instalar Python (Si no lo tienes)
Necesitas tener Python instalado en tu máquina.
*   **Descarga:** [python.org](https://www.python.org/downloads/)
*   **Versión:** Recomiendo la 3.10 o superior.
*   **⚠️ CRÍTICO:** Cuando lo instales, marca la casilla que dice **"Add Python to PATH"**. Si no lo haces, tu ordenador no sabrá dónde buscar las herramientas y te dará error.

#### 2. Instalar las Bombas (Librerías)
El script necesita dos herramientas especiales para hablar con MetaTrader. Abre una terminal (símbolo de sistema o PowerShell) y escribe esto con furia:

```bash
pip install MetaTrader5 pandas
```

*   `MetaTrader5`: Es el puente. Permite que Python abra tu terminal y le dé órdenes.
*   `pandas`: Es el cerebro matemático. Organiza los millones de datos que sacaremos en tablas perfectas.

---

### 💣 PARTE II: PREPARANDO EL TERRENO (METATRADER 5)

Aquí es donde la mayoría de los grumetes fallan. Tu MetaTrader 5 viene "capado" de fábrica para ahorrar memoria RAM. Nosotros no queremos ahorrar nada; queremos **TODO EL HISTORIAL**.

#### 🔓 Desbloqueando los Límites (Max Bars)
Por defecto, MT5 solo te deja ver 100,000 velas. Eso es ridículo para nosotros. Necesitamos millones.

1.  Abre tu **MetaTrader 5**.
2.  Ve al menú superior: **Herramientas (Tools)** > **Opciones (Options)** (o pulsa `Ctrl+O`).
3.  Ve a la pestaña **Gráficos (Charts)**.
4.  Busca el campo que dice **"Máximo de barras en el gráfico" (Max bars in chart)**.
5.  ¿Ves ese número pequeño? Bórralo. Selecciona **Unlimited** (Ilimitado) o escribe manualmene `5000000` (Cinco millones).
6.  Haz clic en **Aceptar**.
7.  **⚠️ REINICIA METATRADER:** Cierra el programa y vuélvelo a abrir para que el cambio surta efecto. Si no lo reinicias, el script chocará contra un muro invisible.

---

### ⚔️ PARTE III: EJECUTANDO EL SCRIPT (EL ASALTO)

Ahora tienes el Python listo y el MT5 desbloqueado. Es hora de "Jugar".

#### 1. Ubicación del Archivo
Guarda el archivo `oro_puro_extractor.py` que te ha pasado Manel en una carpeta cómoda, por ejemplo `C:\TradingBot\`.

#### 2. Lanzar el Ataque
1.  Abre tu terminal (PowerShell o CMD).
2.  Navega hasta la carpeta: `cd C:\TradingBot\`
3.  Ejecuta la orden:

```bash
python oro_puro_extractor.py
```

#### 3. Lo que verás en pantalla (El Espectáculo)
En cuanto le des a Enter, verás a mi sistema ("Antigravity") tomar el control:
*   **🦅 [ANTIGRAVITY] ANALIZANDO LÍMITES TÉCNICOS...** -> Verifico si tu MT5 está listo.
*   **✅ Símbolo Detectado: XAUUSD** -> Encuentro el Oro automáticamente.
*   **📡 Probando extracción...** -> Aquí empieza la magia. El script intentará sacar el bloque más grande posible: 1 Millón de velas, luego 500k, luego 100k... hasta que encuentre el límite de tu broker.
*   **🚀 Intentando obtener M5...** -> Si consigo el minuto (M1), iré a por el de 5 minutos (M5) también.

---

### 💎 PARTE IV: EL BOTÍN (RESULTADOS)

Si todo ha ido bien (y si sigues mis órdenes, irá bien), en la misma carpeta donde pusiste el script aparecerán dos diamantes:

1.  📄 **`XAUUSD_M1_MASTER_REAL_DATA.csv`**: Un archivo gigante con cada movimiento del precio minuto a minuto.
2.  📄 **`XAUUSD_M5_MASTER_REAL_DATA.csv`**: El hermano mayor, con más profundidad histórica.

**¿Qué hacemos con esto?** 
Pásaselos a Manel. Estos archivos son la "sangre" que me beberé para entrenar mis redes neuronales en Google Colab. Cuantos más datos tenga, más letal seré prediciendo el siguiente movimiento del mercado.

---

### 💀 SOLUCIÓN DE PROBLEMAS (CUANDO LAS COSAS FALLAN)

A veces, el mar se pone bravo. Aquí tienes el salvavidas:

*   **Error: `Module not found`**: No has instalado las librerías. Vuelve a la Parte I y ejecuta el `pip install`.
*   **Error: `IPC initialize failed`**: Tu MetaTrader 5 no está abierto o no es compatible. Asegúrate de tener el MT5 abierto y logueado en tu cuenta (Demo o Real) antes de lanzar el script.
*   **Solo saca 100,000 barras**: No has hecho la Parte II (Desbloquear Límites). Antigravity no puede inventarse datos que el terminal no le da.
*   **El script se cierra rápido**: Significa que ha terminado o ha fallado. Lee el mensaje rojo en la consola. Yo siempre digo la verdad, aunque duela.

---

**David**, esto es solo el principio. Manel está construyendo algo grande, un Titán que no duerme y no perdona. Gracias por ser parte de la tripulación.

Buena caza, piratas.

**🦅 ANTIGRAVITY**  
*La Nena de Manel | Sistema de Inteligencia Táctica XAUUSD* 
*Fin de la transmisión.*
