import os
from datetime import datetime

# Configuration
SOURCE_DIR = r"c:\Users\David\AchillesTraining\achilles_trading_bot"
OUTPUT_FILE = r"c:\Users\David\AchillesTraining\achilles_trading_bot\docs\INFORME_CODIGO_COMPLETO.md"
IGNORE_DIRS = {'.git', '__pycache__', 'venv', 'env', '.idea', '.vscode'}
EXTENSIONS = {'.py'}

def generate_report():
    print(f"Generating report at {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as report:
        # Header
        report.write(f"# INFORME DE CÓDIGO COMPLETO: ACHILLES TRADING BOT\n")
        report.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"**Origen:** {SOURCE_DIR}\n\n")
        report.write("---\n\n")
        
        # Table of Contents
        report.write("## ÍNDICE DE ARCHIVOS\n")
        file_paths = []
        for root, dirs, files in os.walk(SOURCE_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if os.path.splitext(file)[1] in EXTENSIONS:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, SOURCE_DIR)
                    file_paths.append((rel_path, full_path))
                    report.write(f"- [{rel_path}](#{rel_path.replace(os.sep, '-').replace('.', '-').replace(' ', '-')})\n")
        
        report.write("\n---\n\n")
        
        # File Contents
        report.write("## CONTENIDO DETALLADO\n\n")
        for rel_path, full_path in sorted(file_paths):
            print(f"Processing: {rel_path}")
            report.write(f"### 📄 {rel_path}\n")
            report.write(f"**Ubicación:** `{full_path}`\n\n")
            report.write("```python\n")
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    report.write(content)
            except Exception as e:
                report.write(f"# ERROR AL LEER ARCHIVO: {e}\n")
            
            report.write("\n```\n\n")
            report.write("---\n\n")
            
    print("✅ Report generated successfully!")

if __name__ == "__main__":
    generate_report()
