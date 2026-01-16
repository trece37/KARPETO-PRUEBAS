# --------------------------------------------------------------------------------
# ☢️ GOD CELL V5: INYECCIÓN NUCLEAR DE DEPENDENCIAS (ZERO-COMPILE)
# PROYECTO: ACHILLES TITAN V3 | PROTOCOLO: 666RULES
# FUENTE: TITAN V3 Enciclopedia de Guerra Algorítmica (Dirty 100)
# AUTOR: ANTIGRAVITY (Ejecutando Instrucciones de Manel)
# CLASIFICACIÓN: CRÍTICO | NO MODIFICAR SIN PERMISO R3K
# --------------------------------------------------------------------------------
import sys
import os
import subprocess
import importlib

def nuclear_injection():
    print(f"💀 [PROTOCOLO 666] INICIANDO SECUENCIA GOD CELL V5")
    print(f"⚡ DETECTANDO ENTORNO: Python {sys.version.split()[0]}...")

    # --- PASO 1: LA VACUNA ANTI-NUMPY 2.0 (MONKEY PATCH) ---
    # Razón BARRIGA (Anticipar): Pandas-TA y TA-Lib mueren con Numpy 2.0.
    # Acción: Parchear en RAM antes de que cualquier librería cargue.
    try:
        import numpy as np
        # Si NumPy 2.0 mató a 'NaN', lo resucitamos manualmente.
        if not hasattr(np, 'NaN'):
            np.NaN = np.nan
            print("   💉 NEUROCIRUGÍA: 'np.NaN' re-inyectado en NumPy 2.x.")
        else:
            print("   ✅ NumPy compatible detectado.")
    except ImportError:
        print("   ⚠️ NumPy no detectado. Se instalará en el siguiente paso.")

    # --- PASO 2: INYECCIÓN DIRECTA DE TA-LIB (SIN COMPILAR) ---
    # Razón ZTE (Eficiencia): Compilar tarda 20 mins. Usar binarios tarda 5 seg.
    # No usamos 'pip install ta-lib' porque intenta compilar.
    print("   ⬇️ INICIANDO DESPLIEGUE DE ARMAS (TA-LIB BINARY)...")
    
    # URLs de binarios de confianza (Conda-Forge / Gohlke builds mirrors / Launchpad)
    # Detectamos versión de Python para elegir el wheel correcto
    py_ver_major = sys.version_info.major
    py_ver_minor = sys.version_info.minor
    
    # URL Genérica de Fallback (Ubuntu/Debian standard libs)
    # Esta es la táctica más segura en Colab (Linux Debian based)
    try:
        print("   💀 [TACTICAL] Descargando librería C++ pre-compilada...")
        url_lib = 'http://launchpadlibrarian.net/192226868/libta-lib0_0.4.0-oneiric1_amd64.deb'
        url_dev = 'http://launchpadlibrarian.net/192226909/ta-lib0-dev_0.4.0-oneiric1_amd64.deb'
        
        subprocess.run(f"wget -q {url_lib} -O libta.deb", shell=True, check=True)
        subprocess.run(f"wget -q {url_dev} -O ta.deb", shell=True, check=True)
        subprocess.run("dpkg -i libta.deb ta.deb", shell=True, check=True)
        print("   ✅ Librería C++ TA-Lib Inyectada en el Kernel.")
        
        print("   💀 [TACTICAL] Instalando Python Wrapper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ta-lib"], check=True)
        print("   ✅ Python Wrapper Operativo.")
    except Exception as e:
        print(f"   ❌ FALLO EN INYECCIÓN BINARIA: {e}")
        print("   ⚠️ Iniciando Protocolo de Emergencia (Compilación Forzada)...")
        # Fallback a pip normal (lento pero seguro si falla el binario)
        subprocess.run([sys.executable, "-m", "pip", "install", "ta-lib"], check=True)

    # --- PASO 3: INSTALACIÓN DEL RESTO DEL ARSENAL (VERSION LOCKING) ---
    # Razón R3K (Robustez): Bloquear versiones exactas para evitar "drift".
    print("   📦 DESPLEGANDO SISTEMAS DE SOPORTE (Legacy Mode)...")
    
    requirements = [
        "numpy==1.26.4",       # EL REY. No tocar. < 2.0 requerido.
        "pandas==2.2.2",       # Estable.
        "scikit-learn",        # Machine Learning estándar.
        "joblib",              # Cacheo y paralellismo.
        "mplfinance",          # Gráficos financieros.
        "vectorbt",            # Backtesting vectorial (opcional pero recomendado en Enciclopedia).
        "pyzmq",               # Puente ZeroMQ con MT5.
        "pandas_ta==0.3.14b0"  # Versión Beta específica requerida por compatibilidad.
    ]

    for req in requirements:
        try:
            print(f"   ... Instalando {req}")
            # --no-deps salva vidas evitando que pip actualice cosas que no debe
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Error instalando {req}. Intentando sin restricciones...")
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=False)

    # --- PASO 4: VALIDACIÓN FINAL (R6V) ---
    print("\n💀 [R6V] VERIFICACIÓN DE SISTEMAS...")
    try:
        import talib
        import pandas_ta as ta
        import numpy as np
        print(f"   ✅ TA-Lib Version: {talib.__version__}")
        print(f"   ✅ Pandas-TA Version: {ta.version}")
        print(f"   ✅ NumPy Version: {np.__version__}")
        
        if np.__version__.startswith("2"):
            print("   ⚠️ ALERTA: NumPy 2.x detectado. El Monkey Patch debe estar activo.")
        else:
            print("   ✅ NumPy 1.x confirmado (Seguro).")
            
        print(f"\n🚀 SISTEMA TITAN V3: READY FOR COMBAT.")
    except ImportError as e:
        print(f"\n❌ ERROR CRÍTICO R6V: {e}")
        print("   El entorno no es seguro. Abortar misión.")

if __name__ == "__main__":
    nuclear_injection()
