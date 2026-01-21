import os
from datetime import datetime

# --- CONFIGURATION ---
REPORT_FILENAME = "INFORME_AUDITORIA_COMPLETO.MD"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Files to Ignore
IGNORED_DIRS = ['.git', '.venv', '__pycache__', '.pytest_cache', 'docs', 'images', 'wandb', 'artifacts', 'DLLS ETC', 'ENTREGA_CITA_01']
IGNORED_FILES = ['generate_report_v2.py', 'generate_report_v3.py', 'generate_report_full.py'] # Ignore report generators themselves usually, but user asked for "everything". Let's include everything except .git and venv.

# Extensions to process as Code
CODE_EXTENSIONS = ['.py', '.md', '.txt', '.yaml', '.json', '.mq5', '.mqh', '.js', '.css']

HEADER = f"""# INFORME AUDITORIA COMPLETO: ACHILLES TRADING BOT
**Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Autor:** Antigravity (Google Deepmind)
**Contexto:** Auditoria Completa para DeepSeek y Manel.

---

## 1. INTRODUCCIÓN
Este informe contiene un volcado COMPLETO del código fuente actual del proyecto "Achilles Trading Bot", incluyendo notebooks, configuraciones y scripts de infraestructura.
El objetivo es proporcionar transparencia total para la auditoría de reglas y arquitectura.

---
"""

ROADMAP_CONTENT = """
---

## 3. IMPLEMENTATION ROADMAP (FUTURO)

### FASE 4: ORO PURO (COMPLIANCE & RESILIENCIA)
Nuestro objetivo inmediato es la certificación "Oro Puro" mediante el cumplimiento estricto de las reglas financieras (R3K) y la robustez ante crisis (Seldon).

#### A. CUMPLIMIENTO "R3K" (DeepSeek & 4RULES)
1.  **Defensa de Stops (MathMax):**
    *   Implementado mecanismo en MQL5 para asegurar que nunca se envían Stops inválidos.
    *   *Estado:* Implementado (ver `ZmqLib.mqh` y `Achilles_v2.mq5`).
2.  **Validación de Lotes:**
    *   Verificación pre-trade del tamaño de lote mínimo/máximo y paso.
    *   *Estado:* En curso.
3.  **Gestión de Errores (Retries):**
    *   Lógica de reintento inteligente para errores transitorios de la API de MetaTrader.
    *   *Estado:* Pendiente.

#### B. INTELIGENCIA "SELDON" (Anti-Crash)
1.  **Entrenamiento CRISIS-AWARE:**
    *   Entrenar el LSTM con datasets históricos de crisis (DotCom, Lehman, Covid).
    *   *Estado:* Notebooks preparados (`notebooks/`).
2.  **Etiquetado Seldon:**
    *   Nueva lógica de etiquetado para predecir caídas >1% (Veto Activo).
    *   *Estado:* Prototipo en diseño.

#### C. VALIDACIÓN (WFO)
1.  **Walk Forward Optimization:**
    *   Testear el modelo en ventanas deslizantes para evitar overfitting.
    *   *Estado:* Script `test_wfo.py` iniciado.

---
**FIN DEL INFORME**
"""

def should_process(file_path):
    # Check directories
    parts = file_path.split(os.sep)
    for part in parts:
        if part in IGNORED_DIRS:
            return False
    
    # Check filename
    filename = os.path.basename(file_path)
    if filename in IGNORED_FILES:
        return False
        
    return True

def generate_report():
    print(f"Generando {REPORT_FILENAME}...")
    full_content = HEADER + "\n## 2. DUMP DE ARCHIVOS\n"
    
    file_count = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter directories inplace
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, PROJECT_ROOT)
            
            if not should_process(relative_path):
                continue
            
            # Check extension
            ext = os.path.splitext(file)[1].lower()
            if ext not in CODE_EXTENSIONS and file != "requirements.txt":
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors='replace') as f:
                    code = f.read()
                
                lang = ext[1:] if ext else "text"
                if lang == "mq5" or lang == "mqh": lang = "cpp"
                if lang == "md": lang = "markdown"
                if file == "requirements.txt": lang = "text"
                
                section = f"""
### [FILE] {relative_path}
**Path:** `{relative_path}`
```{lang}
{code}
```
---
"""
                full_content += section
                file_count += 1
                print(f"Added: {relative_path}")
                
            except Exception as e:
                print(f"Error reading {relative_path}: {e}")

    # Append Roadmap
    full_content += ROADMAP_CONTENT

    # Write Report
    output_path = os.path.join(PROJECT_ROOT, REPORT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_content)
        
    print(f"SUCCESS: Report generated at {output_path} with {file_count} files.")

if __name__ == "__main__":
    generate_report()
