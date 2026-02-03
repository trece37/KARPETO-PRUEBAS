# 🤝 MESA REDONDA - Workspace Colaborativo Multi-IA

## 📋 Resumen Ejecutivo

Este documento explica el **workflow colaborativo** diseñado para permitir que múltiples IAs (Perplexity, Gemini, Claude, Antigravity, etc.) trabajen juntas en proyectos compartidos **sin necesidad de edición manual** por parte del usuario.

---

## 🎯 El Problema Original

**Necesidad:** Un espacio común donde:
- Múltiples IAs puedan **leer** contenido
- Múltiples IAs puedan **escribir** contenido bajo orden del usuario
- El espacio esté **indexado en Google** para acceso universal
- **Sin OAuth complejo** ni commits manuales engorrosos
- El usuario actúa como **orquestador** sin editar manualmente

**Descartado:**
- ❌ Google Drive → OAuth inviable para muchas IAs
- ❌ GitHub con commits manuales → Demasiado liante

---

## ✅ La Solución: Arquitectura Híbrida

### 🏗️ Componentes del Sistema

```
┌─────────────────────────────────────────────────────────┐
│          ESPACIO COMPARTIDO (Dual Backend)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📁 Google Drive                 🐙 GitHub Gist        │
│  (Carpeta Pública)               (Público)             │
│  ↓                               ↓                     │
│  Antigravity escribe aquí        Espejo público        │
│  (Drive API OAuth local)         (Todas las IAs leen)  │
│                                                         │
└─────────────────────────────────────────────────────────┘
         ↑ ESCRIBE                    ↑ LEEN
         │                            │
    ┌────┴────┐              ┌────────┴────────┐
    │         │              │                 │
┌───▼────┐    │      ┌───────▼──┐  ┌──────▼────┐
│Antigrav│    │      │Perplexity│  │  Gemini   │
│ (local)│    │      │  (web)   │  │   (web)   │
└────────┘    │      └──────────┘  └───────────┘
              │
          ┌───▼────┐
          │ Claude │
          │ (web)  │
          └────────┘
```

---

## 🔄 Workflow Paso a Paso (Sin Edición Manual)

### Setup Inicial (Una sola vez)

1. **Crear carpeta en Drive**
   - Nombre: `AI-Shared-Workspace`
   - Permisos: `Anyone with link can view`
   - URL: `https://drive.google.com/drive/folders/[ID]`

2. **Crear GitHub Gist público**
   - Nombre: `manel-ai-workspace.md`
   - Contenido: Estructura base con secciones para cada IA
   - URL: `https://gist.github.com/trece37/[ID]`

3. **Configurar Antigravity (local)**
   - OAuth conectado a Google Drive
   - Personal Access Token de GitHub (scope: `gist`)
   - Comandos habilitados para subir/actualizar archivos

---

### Escenario 1: Perplexity Analiza → Antigravity Guarda

**Flujo:**
1. Usuario → Perplexity: *"Analiza el estado de Achilles LSTM y genera reporte técnico"*
2. Perplexity → Usuario: *[Genera análisis completo con métricas, patterns, etc.]*
3. Usuario → Antigravity: *"Toma este análisis de Perplexity y guárdalo en:*
   - *Drive: `AI-Shared-Workspace/perplexity-analisis-2026-02-03.md`*
   - *Gist: Actualiza sección Perplexity"*
4. Antigravity ejecuta:
   ```python
   # Sube a Drive vía Google Drive API
   drive.files().create(body={'name': 'perplexity-analisis-2026-02-03.md', 
                               'parents': ['[FOLDER_ID]']}, 
                        media_body=content)
   
   # Actualiza Gist vía GitHub API
   requests.patch('https://api.github.com/gists/[GIST_ID]',
                  headers={'Authorization': f'token {GITHUB_TOKEN}'},
                  json={'files': {'manel-ai-workspace.md': {'content': updated_content}}})
   ```
5. **Resultado:** Contenido disponible en Drive y Gist **sin tocar nada manualmente**

---

### Escenario 2: Colaboración Multi-IA (El Santo Grial)

**Flujo Completo:**

1. **Usuario → Perplexity:**
   *"Analiza datos técnicos de XAUUSD últimas 48h: patterns, soporte/resistencia, ML signals"*

2. **Perplexity genera:**
   - Análisis técnico ML
   - Detección de patterns (Head & Shoulders, etc.)
   - Niveles clave

3. **Usuario → Antigravity:**
   *"Guarda análisis de Perplexity en workspace"*
   - Antigravity sube a Drive + actualiza Gist

4. **Usuario → Gemini:**
   *"Lee el Gist https://gist.github.com/trece37/[ID] y añade análisis fundamental: noticias Fed, datos empleo, geopolítica"*

5. **Gemini genera:**
   - Análisis fundamental
   - Correlaciones macro
   - Riesgos geopolíticos

6. **Usuario → Antigravity:**
   *"Añade análisis de Gemini al workspace"*
   - Antigravity actualiza Drive + Gist

7. **Usuario → Perplexity:**
   *"Lee el workspace completo y haz síntesis final combinando análisis técnico + fundamental. Dame tu decisión de trading"*

8. **Perplexity lee Gist actualizado y sintetiza:**
   - Combina ambos análisis
   - Detecta confluencias
   - Recomienda estrategia

**Resultado:** Análisis 360° sin editar nada tú, solo orquestando.

---

## 🔧 Comandos Clave de Antigravity

### Para Drive
```bash
# Subir archivo nuevo
"Sube [archivo] a Drive en [ruta]"

# Actualizar archivo existente
"Actualiza [archivo] en Drive con [contenido]"

# Leer archivo
"Lee [archivo] de Drive en [ruta] y procesa"
```

### Para GitHub Gist
```bash
# Actualizar Gist completo
"Actualiza Gist [ID] con [contenido]"

# Añadir sección a Gist
"Añade a Gist [ID] la sección [nombre] con [contenido]"

# Leer Gist
"Lee Gist [ID] y analiza"
```

---

## 📖 Acceso de Lectura (Todas las IAs)

### Perplexity, Gemini, Claude, etc.

**GitHub Gist (Preferido):**
```
URL: https://gist.github.com/trece37/[ID]
- Público
- Indexado en Google
- Markdown renderizado
- Historial de versiones automático
```

**Google Drive:**
```
URL: https://drive.google.com/file/d/[ID]/view
- Público con link
- Puede requerir conversión a texto plano
- Mejor para archivos grandes (datasets, CSVs)
```

**Comando al usuario:**
*"Lee https://gist.github.com/trece37/[ID] y dime qué ha dicho Gemini en su última sección"*

---

## 🚀 Automatización Nivel PRO (Opcional)

### Script de Sincronización `workspace_sync.py`

```python
#!/usr/bin/env python3
"""
Orquestador automático Drive ↔ Gist
Escucha carpeta local, sincroniza con Drive y Gist
"""

import os
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GIST_ID = os.getenv('GIST_ID')
DRIVE_FOLDER_ID = os.getenv('DRIVE_FOLDER_ID')

class WorkspaceSyncHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Leer archivo modificado
        with open(event.src_path, 'r') as f:
            content = f.read()
        
        # Subir a Drive
        self.upload_to_drive(event.src_path, content)
        
        # Actualizar Gist
        self.update_gist(event.src_path, content)
    
    def upload_to_drive(self, filepath, content):
        # Implementación Drive API
        pass
    
    def update_gist(self, filepath, content):
        filename = os.path.basename(filepath)
        url = f'https://api.github.com/gists/{GIST_ID}'
        headers = {'Authorization': f'token {GITHUB_TOKEN}'}
        data = {'files': {filename: {'content': content}}}
        requests.patch(url, headers=headers, json=data)

if __name__ == '__main__':
    observer = Observer()
    observer.schedule(WorkspaceSyncHandler(), path='./workspace', recursive=True)
    observer.start()
    print('🚀 Workspace sync activo...')
    observer.join()
```

**Uso:**
```bash
# Usuario a Antigravity:
"Ejecuta workspace_sync.py con el análisis de Perplexity"

# Antigravity ejecuta script → todo se sincroniza automáticamente
```

---

## 🎓 Casos de Uso Reales

### Trading Bot Achilles
- **Perplexity:** Análisis técnico ML, backtesting
- **Gemini:** Análisis fundamental, noticias macro
- **Claude:** Revisión de código, optimizaciones
- **Antigravity:** Orquestación, guardado automático
- **Usuario:** Toma decisión final basada en consenso

### Desarrollo de Proyectos
- **Cada IA aporta su especialidad**
- **Workspace central con historial completo**
- **Usuario revisa y aprueba sin copiar/pegar**

### Research Multi-Perspectiva
- **Cada IA investiga un ángulo diferente**
- **Síntesis final combina todos los insights**
- **Documentación automática en Drive + Gist**

---

## 📝 Ventajas de Esta Arquitectura

✅ **Cero Edición Manual:** El usuario solo orquesta, no copia/pega  
✅ **Historial Completo:** GitHub Gist guarda todas las versiones  
✅ **Acceso Universal:** Cualquier IA puede leer vía URL pública  
✅ **Escritura Controlada:** Solo Antigravity escribe (evita pisadas)  
✅ **Escalable:** Añadir más IAs es trivial  
✅ **Backup Dual:** Drive + GitHub = redundancia  
✅ **Indexado Google:** Búsquedas futuras encuentran el contenido  

---

## 🔮 Próximos Pasos

1. **Implementar estructura base de Gist** con secciones definidas
2. **Testear comandos de Antigravity** para Drive + GitHub
3. **Crear plantillas markdown** para cada tipo de análisis
4. **Definir convenciones de nombres** de archivos (`YYYY-MM-DD-[IA]-[tema].md`)
5. **Montar script de sincronización automática** (opcional)
6. **Integrar con notificaciones** (Telegram/Discord) cuando una IA termine su parte

---

## 🤖 Metadata

**Creado:** 2026-02-03  
**Autor:** Manel (trece37) + Perplexity  
**Propósito:** Workspace colaborativo multi-IA sin edición manual  
**Stack:** Google Drive + GitHub Gist + Antigravity (local)  
**Estado:** Diseño completo, pendiente implementación  

---

## 📚 Referencias

- [Conversación original completa](attached_file:1)
- [Google Drive API Docs](https://developers.google.com/drive/api/v3/about-sdk)
- [GitHub Gist API](https://docs.github.com/en/rest/gists)
- [Antigravity Setup](https://github.com/google/antigravity)

---

**¡LA MESA REDONDA ESTÁ SERVIDA! 🍷🤝**
