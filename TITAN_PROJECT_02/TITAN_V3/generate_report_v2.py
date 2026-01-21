import os

# Output File
OUTPUT_FILE = "../../INFORME_JARVIS.02_FASE2.MD"

# Files to Include (Relative to achilles_trading_bot root)
FILES = [
    ("CONTEXT", "../../ENTREGA_CITA_01/JARVIS_CORE_PROTOCOL.md"),
    ("CONTEXT", "../../ENTREGA_CITA_01/INSTRUCCIONES_CONTEXTO.md"),
    ("TASK", "../../.gemini/antigravity/brain/dd1fd7c8-eea2-45f3-881d-5425bbd4cb79/task.md"),
    ("API", "src/brain/api/main.py"),
    ("MODELS", "src/brain/models/seldon.py"),
    ("MODELS", "src/brain/models/roi_alpha.py"),
    ("MODELS", "src/brain/models/lstm.py"),
    ("MODELS", "src/brain/models/portfolio.py"),
    ("STRATEGY", "src/brain/strategy/protection.py"),
    ("STRATEGY", "src/brain/strategy/roi.py"),
    ("CORE", "src/brain/core/types.py"),
    ("CORE", "src/brain/core/interfaces.py"),
    ("WORKER", "src/worker/Experts/Achilles_v1.mq5"),
    ("CONFIG", "requirements.txt"),
    ("VALIDATION", "verify_veto.py")
]

def generate():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Header
        out.write("# INFORME JARVIS 02: FASE 2 COMPLETADA (CODE DUMP)\n")
        out.write(f"Generated Audit Report for Gemini 3. Contains full source code after Phase 2 fixes.\n\n")
        
        for category, path in FILES:
            out.write(f"\n## {category}: {os.path.basename(path)}\n")
            out.write(f"path: `{path}`\n\n")
            
            ext = path.split('.')[-1]
            lang = "python"
            if ext == "mq5": lang = "cpp"
            if ext == "md": lang = "markdown"
            if ext == "txt": lang = "text"
            
            if os.path.exists(path):
                out.write(f"```{lang}\n")
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                        out.write(content)
                except Exception as e:
                    out.write(f"Error reading file: {e}")
                out.write("\n```\n")
            else:
                out.write(f"> [!WARNING]\n> File not found: {path}\n")
                
    print(f"Report generated at {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    generate()
