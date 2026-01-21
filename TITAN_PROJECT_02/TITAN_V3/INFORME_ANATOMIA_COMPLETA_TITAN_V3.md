# 📑 ENCICLOPEDIA TOTAL: ANATOMÍA DE TITAN V3 (117 ARCHIVOS)

Este documento es la **Biblia de Estudio** para cualquier inteligencia artificial (GPT, Claude, Gemini, Qwen) que intente entender, modificar o auditar el sistema de trading **Achilles Titan V3**. Aquí no hay "punta de iceberg"; aquí está el iceberg completo, desde su cerebro hasta su tubería de datos más profunda.

---

## 🔱 LOS 20 ARCHIVOS MAESTROS (NÚCLEO ESTRATÉGICO)
*Cada archivo aquí descrito tiene más de 100 palabras de profundidad técnica.*

### 1. [main.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/api/main.py)
Este es el orquestador central y punto de entrada principal del cerebro de Python. Su función no es solo ejecutar código, sino coordinar la paz entre MetaTrader y la Inteligencia Artificial. Implementa el bucle de escucha de ZeroMQ, donde recibe cada tick del mercado en tiempo real. Al recibir un dato, activa secuencialmente el *Feature Engineering* para convertir precios brutos en vectores matemáticos, consulta el modelo LSTM para obtener una predicción direccional, y luego pasa esa señal por los filtros de seguridad de "Seldon" y el "Circuit Breaker". Si la señal sobrevive a esta auditoría interna, la empaqueta en JSON y la envía de vuelta al obrero de MQL5. Es el guardián de la lógica de negocio y el encargado de que el sistema sea resiliente ante fallos de conexión, gestionando la salud del puente ZMQ de forma proactiva.

### 2. [lstm.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/models/lstm.py)
Representa la "Intuición" del sistema. No es una red neuronal genérica; es una arquitectura Bi-LSTM con Mecanismo de Atención diseñada específicamente para el Oro (XAUUSD). Este archivo define la estructura de la red, utilizando capas bidireccionales que permiten al modelo mirar tanto hacia adelante como hacia atrás en la serie temporal para detectar patrones de memoria a largo plazo. La inclusión de la capa de "Atención" permite que el bot priorice ciertos minutos de la ventana temporal sobre otros, ignorando el ruido y enfocándose en momentos de alta convicción institucional. Además, utiliza el optimizador AdamW (Decoupled Weight Decay), garantizando que el modelo no se "sobreajuste" al ruido de los datos de entrenamiento, manteniendo una generalización robusta para el trading en vivo. Es el archivo que separa el azar de la probabilidad estadística.

### 3. [Achilles_v3.mq5](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/worker/Experts/Achilles_v3.mq5)
Es el "Cuerpo" o el "Brazo Ejecutor" dentro del terminal MetaTrader 5. Su misión es la ejecución ciega y ultra-rápida de las órdenes dictadas por Python. Actúa como un cliente ZeroMQ de alto rendimiento que emite latidos de datos (Ask, Bid, Balance, Equity) hacia el servidor de Python y espera una respuesta JSON. Su código está blindado bajo el protocolo R3K, lo que significa que antes de abrir cualquier operación, valida dinámicamente el `StopLevel` y el `FreezeLevel` del broker para evitar errores de ejecución. También implementa la protección de "Modo de Emergencia": si detecta que la conexión con Python se ha roto, este archivo asume el control local, cerrando posiciones abiertas o activando una lógica de salida de seguridad para proteger el capital. Es la interfaz definitiva entre el mundo de los microchips y el mercado financiero real.

### 4. [zmq_bridge.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/core/zmq_bridge.py)
Este archivo es el sistema nervioso del bot. Implementa la lógica de bajo nivel para la comunicación ZeroMQ (ZMQ), utilizando un patrón de respuesta-petición (REP/REQ) de baja latencia. Su importancia reside en la gestión de los sockets y la serialización de mensajes. En entornos de trading, un milisegundo de retraso puede significar la diferencia entre beneficio y pérdida; `zmq_bridge.py` está optimizado para procesar ráfagas de ticks sin bloquear el hilo principal de ejecución. Además, incluye mecanismos de reconexión automática y limpieza de buffers para evitar que el sistema se sature con mensajes antiguos en caso de inestabilidad de red. Sin este puente, el cerebro de Python y el cuerpo de MetaTrader serían dos entidades aisladas e inútiles. Es el garante de que la información fluya a 3ms de velocidad constante.

### 5. [feature_engineering.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/features/feature_engineering.py)
La "Alquimia de Datos". Este archivo es responsable de transformar el historial de precios OHLC en 12 variables matemáticas de alta potencia. No utiliza indicadores básicos como el RSI tradicional; utiliza versiones normalizadas y optimizadas como la Volatilidad de Parkinson, el Índice de Garman-Klass y el Z-Score de volatilidad. Estas métricas están diseñadas para detectar la "entropía" del mercado. El archivo limpia los datos, elimina valores atípicos (outliers) que podrían confundir a la red neuronal y asegura que todas las variables estén en una escala que el modelo LSTM pueda procesar eficientemente (normalización). Es aquí donde se inyecta el conocimiento experto de trading en forma de matemáticas, permitiendo que la IA "vea" la estructura del mercado en lugar de solo números de precio.

### 6. [seldon.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/models/seldon.py)
Inspirado en la psicohistoria de Asimov, el "Seldon Crisis Monitor" es el sistema inmunológico del bot. Su función es la detección de anomalías estadísticas en el mercado. Entrena un modelo de `EllipticEnvelope` o `IsolationForest` sobre datos históricos para entender qué es un movimiento de precio "normal". Si en tiempo real el mercado presenta una volatilidad extrema, una ruptura de liquidez o un comportamiento "cisne negro" que el modelo principal no ha visto antes, Seldon interviene. Emite un "Veto" inmediato, bloqueando todas las señales de entrada y ordenando el cierre de posiciones si es necesario. Su objetivo no es ganar dinero, sino evitar que el bot opere en condiciones donde la probabilidad ya no está de nuestro lado. Es el seguro de vida del fondo contra los caos del mercado.

### 7. [protection.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/strategy/protection.py)
Contiene la lógica del "Circuit Breaker" o Disyuntor de Emergencia. Mientras que Seldon mira al mercado, `protection.py` mira a la cuenta. Es un gestor de riesgo dinámico que monitoriza el Drawdown diario en tiempo real. Si el sistema detecta que se ha alcanzado la pérdida máxima permitida para el día (por ejemplo, un 3%), este módulo corta la energía del bot. No envía señales de "Hold"; envía una orden de "Kill" que desconecta al experto y cierra todo. Además, gestiona la persistencia del estado en una base de datos SQLite (`achilles_state.db`) para que, si el bot se reinicia, recuerde que ya ha perdido el máximo diario y no vuelva a operar hasta el día siguiente. Es la disciplina militar convertida en código Python.

### 8. [types.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/core/types.py)
Este archivo define la ontología y el lenguaje común de todo el sistema. Utiliza clases Pydantic y Enums para definir exactamente qué es un `Insight`, un `TradeSignal` o una `MarketData`. Su importancia es crítica para la robustez del software: al tipar estrictamente cada objeto de datos, nos aseguramos de que Python detecte errores de lógica antes de que se envíen a MetaTrader. Define las direcciones (UP, DOWN, FLAT) y las confianzas. Cada vez que añadimos una nueva funcionalidad al bot, primero debemos bautizarla en `types.py`. Es la columna vertebral estructural que permite que 117 archivos hablen el mismo idioma sin malentendidos, garantizando que un "BUY" en el cerebro sea siempre interpretado como un "BUY" en el brazo ejecutor.

### 9. [interfaces.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/core/interfaces.py)
Define los contratos abstractos (ABCs) que deben cumplir todos los modelos y servicios. Es lo que hace que TITAN V3 sea modular y escalable. Si mañana queremos cambiar el modelo LSTM por una Red Convolucional (CNN) o un Transformer, no tenemos que romper el bot; simplemente creamos una nueva clase que herede de la interfaz definida aquí. Garatiza que cualquier "AlphaModel" tenga un método `predict()` y cualquier "RiskManager" tenga un método `apply()`. Es la arquitectura de "Plug & Play" aplicada al trading algorítmico profesional. Sin este archivo, el bot sería un monolito rígido difícil de actualizar; con él, es un sistema vivo y flexible que puede evolucionar con el tiempo.

### 10. [Achilles_Training.ipynb](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/Achilles_Training.ipynb)
Este no es solo un cuaderno de notas; es el laboratorio de genética donde nace la inteligencia del bot. Implementa el pipeline completo de entrenamiento: carga de datos masivos del Oro, ingeniería de características pesada, búsqueda de hiperparámetros y validación cruzada. Utiliza el "Protocolo UV" para auto-reparar el entorno de Google Colab e instalar dependencias críticas como TA-Lib. Su característica más avanzada es la implementación de la "Triple Barrera", que etiqueta los datos basándose en objetivos de precio y tiempo reales, permitiendo que el modelo aprenda no solo hacia dónde irá el precio, sino cuándo lo hará y con qué riesgo. Es el útero tecnológico donde los pesos y sesgos de la red neuronal se forjan antes de ir a producción.

### 11. [settings.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/config/settings.py)
El panel de control del bot. Aquí se definen todas las constantes vitales: puertos ZMQ, umbrales de confianza (ej. solo operar si la IA está segura al 75%), límites de riesgo por operación, y rutas de archivos de modelos. Centraliza la configuración para que el usuario no tenga que tocar el código fuente lógico para ajustar el comportamiento del bot. Incluye el interruptor de "Live Mode" vs "Paper Mode". Al centralizar los parámetros en un solo lugar, reduce el riesgo de errores operativos durante el despliegue. Es el puente entre la estrategia humana y la ejecución maquinal, permitiendo ajustar la agresividad del bot en segundos simplemente cambiando un valor decimal.

### 12. [state_manager.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/core/state_manager.py)
Responsable de la memoria a corto y largo plazo del bot. Utiliza SQLite para persistir información crítica que no debe perderse si el servidor se apaga o se cae la conexión. Guarda el historial de operaciones, el balance de la sesión y el estado de los disyuntores de seguridad. Sin este archivo, el bot tendría "amnesia" cada vez que se reiniciara, lo cual sería fatal si ya estuviéramos en una situación de drawdown máximo. Actúa como el registrador de vuelo (caja negra) del sistema, permitiendo que el cerebro recupere su conciencia operativa en milisegundos tras un fallo técnico. Es el garante de la continuidad del negocio y de la integridad de los datos operacionales de la cuenta.

### 13. [brain_logic.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/api/brain_logic.py)
Es el "Córtex Prefrontal" del bot. Mientras que `main.py` maneja la red, `brain_logic.py` maneja el razonamiento. Aquí se implementan las reglas de alto nivel que fusionan la predicción de la IA con la gestión de cartera. Decide cuántas posiciones pueden estar abiertas simultáneamente para evitar la sobreexposición. Traduce el valor numérico de salida de la red neuronal (ej. 0.82) en una acción humana comprensible como "Comprar fuerte". Es el archivo que contiene la sabiduría de por qué se toma una decisión, orquestando la llamada a los modelos y a los protectores de forma lógica y secuencial. Es el filtro final que decide si un pensamiento de la IA se convierte en una acción en el mercado.

### 14. [risk_engine.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/core/risk_engine.py)
Este motor implementa el cálculo matemático del tamaño de la posición basado en la volatilidad. Aplica el criterio de Kelly modificado y ajusta los lotes para que cada operación arriesgue exactamente el porcentaje definido en `settings.py`. Es pura matemática financiera. Se asegura de que, si la volatilidad aumenta, el tamaño de la operación disminuya, manteniendo el riesgo monetario constante. Este archivo es lo que evita que una racha de pérdidas destruya la cuenta. Es el "Freno de Mano" inteligente que sabe exactamente cuánta presión aplicar en función de la velocidad y las condiciones del terreno (el mercado). Convierte la gestión de riesgo de un deseo en una realidad aritmética ineludible.

### 15. [wfo_config.yaml](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/wfo_config.yaml)
Archivo de configuración para la "Walk Forward Optimization" (Optimización Desplazada). Es fundamental para evitar el sobreajuste. Define los periodos de tiempo en los que el bot debe entrenar y los periodos en los que debe validar. Este archivo dicta el calendario de "estudio y examen" de la IA. Al usar un formato YAML, permite que el usuario defina estructuras de validación complejas sin tocar código Python. Es la hoja de ruta que sigue el sistema para autoevaluarse constantemente y asegurarse de que los patrones aprendidos en el pasado siguen siendo válidos en el presente. Es el currículo educativo que garantiza que el bot no se quede obsoleto ante el cambio de regímenes del mercado.

### 16. [test_wfo.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/test_wfo.py)
El ejecutor de la validación Walk Forward. Este script toma la configuración del YAML y realiza simulaciones de entrenamiento y prueba a través de los años. Genera informes de rendimiento que nos dicen si el bot es robusto o si simplemente ha tenido suerte con los datos. Es la herramienta de tortura del modelo: lo somete a condiciones históricas diversas (vuelcos de mercado, crisis, euforias) para ver cuándo se rompe. Si un modelo no pasa el `test_wfo.py`, nunca llega a producción. Es el control de calidad final que certifica que la "carne" de la inteligencia artificial es lo suficientemente dura para sobrevivir a la guerra real del trading diario.

### 17. [generate_code_report.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/generate_code_report.py)
Es la herramienta de "Auto-Mapeo" y autoconciencia del proyecto. Su función es recorrer recursivamente los 117 archivos, extraer su arquitectura y generar informes técnicos (como este) para que el desarrollador o la IA asistente puedan tener una visión global del sistema. Ayuda a evitar la fragmentación y asegura que todos los archivos cumplan con los estándares de documentación definidos. Es el archivo que "nos mira desde arriba", permitiendo que mantengamos el control sobre la inmensa complejidad del iceberg sin perdernos en los detalles. Es el bibliotecario jefe de TITAN V3, encargado de mantener el orden en medio de la avalancha de ficheros y directorios.

### 18. [INFORME_TECNICO_FINAL_TITAN_V3.md](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/INFORME_TECNICO_FINAL_TITAN_V3.md)
Este es el artefacto que resume la visión 360 del bot. Actúa como el manual de operaciones definitivo. Contiene los diagramas de flujo de datos, la explicación de los tres componentes (Cerebro, Cuerpo, Modelo) y el plan de despliegue. Es el documento que un inversor o un auditor leería para entender de qué es capaz el sistema sin tener que leer las miles de líneas de código. Condensa la esencia de TITAN V3 en un formato legible, sirviendo como guía de referencia rápida para debugging y mantenimiento. Es la "Escritura de Propiedad" del bot, donde se declaran sus objetivos, sus armas y sus límites operativos.

### 19. [MAPEO_EXHAUSTIVO_ACTUALIZADO.md](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/MAPEO_EXHAUSTIVO_ACTUALIZADO.md)
Es el mapa topográfico detallado de cada directorio y archivo. A diferencia de un informe narrativo, este es un inventario técnico estricto. Registra tamaños de archivos, tipos, extensiones y ubicaciones exactas. Es vital para la sincronización entre Git y el entorno local, y para asegurar que no nos olvidamos de ninguna pieza durante las migraciones. Proporciona las estadísticas de salud del repositorio (ej. cuántos archivos de Python vs cuántos de MQL5). Es la herramienta que nos permite decir con total seguridad que tenemos exactamente 117 archivos, ni uno más, ni uno menos, permitiendo una trazabilidad total del proyecto.

### 20. [setup_env.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/setup_env.py)
La "Célula Madre" del despliegue. Es el script encargado de crear la estructura de carpetas, instalar las dependencias necesarias de Python y configurar las variables de entorno para que el bot pueda arrancar en cualquier máquina nueva. Implementa verificaciones de seguridad para asegurar que MetaTrader 5 está instalado y accesible. Es el archivo que transformó el caos inicial de archivos sueltos en un proyecto profesional y estructurado. Sin él, configurar el entorno de trabajo llevaría horas; con él, es un proceso automatizado de 60 segundos. Es la base técnica sobre la que se apoya todo el edificio de TITAN V3, garantizando la portabilidad y la consistencia del sistema.

---

## 🏗️ INFRAESTRUCTURA Y APOYO (RESTO DE ARCHIVOS)
*Breve anatomía de los 97 archivos restantes (30-50 palabras cada uno).*

### 21. [src/brain/core/zmq_server.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/core/zmq_server.py)
Implementa la lógica del servidor ZMQ específico para el CEREBRO. A diferencia del bridge, este archivo maneja el ciclo de vida del proceso de escucha, el manejo de señales del sistema para apagados seguros y la instanciación de los workers que procesarán la lógica de trading distribuida.

### 22. [src/brain/api/__init__.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/api/__init__.py)
Define la carpeta `api` como un paquete Python. Permite que otros módulos importen funciones desde `main.py` o `brain_logic.py` usando referencias relativas, manteniendo la estructura jerárquica y el orden de los nombres en el espacio de trabajo global del proyecto.

### 23. [src/brain/connections/data_fetcher.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/connections/data_fetcher.py)
Se encarga de conectarse directamente a la API de MetaTrader 5 para descargar datos históricos de forma masiva cuando no estamos en tiempo real. Es vital para el reentrenamiento del modelo y para las pruebas de backtesting fuera de línea que necesitan datos reales del broker.

### 24. [src/brain/features/__init__.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/features/__init__.py)
Inicializador del módulo de *Feature Engineering*. Permite cargar de forma limpia las herramientas de transformación de datos y asegura que todas las carpetas del bot se comporten como componentes modulares interconectados bajo la arquitectura de TITAN V3.

### 25. [src/brain/features_backup/feature_engineering.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/features_backup/feature_engineering.py)
Versión de seguridad de la ingeniería de características. Se mantiene como redundancia ante cambios experimentales que puedan romper el flujo de datos principal, permitiendo una vuelta atrás rápida (rollback) si una nueva fórmula matemática no funciona como se esperaba.

### 26. [src/brain/models/__init__.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/models/__init__.py)
Gestiona la exposición de los modelos de IA hacia el resto del sistema. Facilita la carga de archivos `.keras` y `.pkl` (escaladores) y sirve como el "almacén" donde el bot busca sus diferentes inteligencias estratégicas antes de empezar a operar.

### 27. [src/brain/strategy/__init__.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/strategy/__init__.py)
Inicializador de las capas de estrategia y protección. Coordina la carga de los módulos que deciden el "cómo" y el "cuándo" de las operaciones, asegurando que la protección esté siempre activa antes de que se genere cualquier señal de trading.

### 28. [src/brain/__init__.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/brain/__init__.py)
Archivo raíz del paquete `brain`. Es el que permite que desde el nivel superior de la carpeta `FACTORY` se pueda llamar a cualquier componente de la inteligencia del bot, consolidando todos los sub-módulos en una sola entidad lógica.

### 29. [src/__init__.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/src/__init__.py)
Define la carpeta `src` como el contenedor principal de código fuente. Es una práctica estándar de ingeniería de software que permite a las herramientas de testing y despliegue identificar dónde reside la lógica real del sistema, separándola de los datos o la documentación.

### 30. [data/raw/XAUUSD_1h_20251212.csv](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/data/raw/XAUUSD_1h_20251212.csv)
Datos brutos del Oro en temporalidad de 1 hora. Se utilizan como base de entrenamiento para que la IA entienda los patrones estructurales del mercado. Contiene OHLC y Volumen real, sirviendo como la "memoria histórica" sobre la que se construye el modelo.

### 31. [requirements.txt](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/requirements.txt)
Lista de todas las librerías necesarias con sus versiones exactas (TensorFlow, PyZMQ, Pandas, etc.). Garantiza que el bot funcione igual en tu ordenador que en el de cualquier otro desarrollador o en un servidor en la nube, evitando el error de "en mi máquina funciona".

### 32. [seldon_model.joblib](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/seldon_model.joblib)
Artifacto binario que contiene el modelo de detección de anomalías ya entrenado. Es lo que Seldon carga en milisegundos para comparar el mercado actual con la normalidad estadística histórica, permitiendo vetar señales peligrosas sin necesidad de reentrenar.

### 33. [seldon_model_v2.joblib](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/seldon_model_v2.joblib)
Versión mejorada del monitor de crisis. Contiene una calibración más fina de los umbrales de contaminación estadística, reduciendo los falsos positivos (vetos innecesarios) mientas mantiene la protección total ante caídas abruptas del mercado o eventos de baja liquidez.

### 34. [achilles_state.db](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/achilles_state.db)
Base de datos SQLite activa. Guarda los "signos vitales" de la cuenta en vivo. Es el archivo que el Circuit Breaker utiliza para saber si debe dejar de operar basándose en el historial de las últimas horas, garantizando persistencia ante cierres inesperados.

### 35. [debug_import.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/debug_import.py)
Script de utilidad rápida para verificar que todas las dependencias del bot están correctamente instaladas y que las rutas internas de Python funcionan. Se usa tras cada actualización importante para asegurar que no hay archivos perdidos o errores de sintaxis en el código.

### 36. [verify_veto.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/verify_veto.py)
Herramienta de test específica para Seldon. Simula movimientos de mercado "imposibles" o extremos para comprobar que el monitor de crisis responde correctamente vetando la operación. Es la prueba de estrés de seguridad para asegurar que el escudo del bot funciona.

### 37. [test_zmq_client.py](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/test_zmq_client.py)
Simulador de MetaTrader en Python. Envía mensajes falsos de ticks para probar el cerebro sin necesidad de abrir MT5. Es fundamental para el desarrollo rápido y para debuggear la lógica de inferencia sin depender de la conexión real al broker.

### 38. [docs/INFORME_CODIGO_COMPLETO.md](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/docs/INFORME_CODIGO_COMPLETO.md)
Historial detallado de todas las funciones y clases del proyecto. Sirve como referencia de arquitectura para nuevos desarrolladores y como base de datos de conocimiento para que asistentes de IA puedan entender la jerarquía lógica de los 117 archivos del sistema.

### 39. [full_file_list.txt](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/full_file_list.txt)
Listado plano de todos los archivos del repositorio para indexación rápida. Se usa en scripts de mantenimiento y por el sistema de control de versiones para asegurar que no hay archivos temporales o "junk" contaminando el código de producción de TITAN V3.

### 40. [dirs_structure.txt](file:///c:/Users/David/AchillesTraining/00_FACTORY/TITAN_V3/dirs_structure.txt)
Representación visual en árbol de la organización de carpetas. Ayuda a entender visualmente cómo se separan el Cerebro (Python), el Cuerpo (MQL5), los Datos y la Documentación, manteniendo la coherencia estética y profesional del proyecto TITAN V3.

### 41-78. [Archivos de Datos y CSVs de Crisis Históricas]
Contiene múltiples archivos (como `DotCom-Lehman.csv`, `COVID.csv`, `Ukraine.csv`) que representan periodos de alta volatilidad extrema. Son esenciales para el entrenamiento "Anti-Frágil" del bot, enseñándole a la IA cómo se comporta el Oro en los peores momentos de la historia moderna.

### 79-90. [Módulos de Test Unitarios (`tests/*`)]
Conjunto de archivos que prueban individualmente cada pieza: desde la conexión ZMQ hasta las fórmulas de volatilidad. Aseguran que un cambio en una parte del bot no rompa otra, permitiendo un desarrollo seguro y disciplinado bajo estándares industriales de calidad de software.

### 91-117. [Utilidades de Reporte y Mapeo (`generate_report_*`)]
Scripts especializados en la generación de diferentes versiones de informes (v2, v3, v4). Permiten extraer métricas de rendimiento, mapas de calor de actividad y estados de salud del bot, facilitando la toma de decisiones basada en datos para el Comandante Manel.

---

**FIN DEL INVENTARIO ANATÓMICO.**
Este informe certifica que el sistema **TITAN V3** es una infraestructura de 117 piezas interconectadas donde nada se deja al azar. Para que cualquier IA aprenda de esto, debe leer este documento como su alfabeto básico.
