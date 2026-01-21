# GUÍA DE EJECUCIÓN: DATA PIPELINE & TRAINING
**Autor:** Antigravity  
**Fase:** 2. ORO PURO

---

## 1. PREPARACIÓN DE DATOS ("DATA DATASETS")
**Objetivo:** Asegurar que los datos CSV estén en la carpeta correcta de Google Drive.

1.  **Localiza tus archivos CSV** (Local):
    *   Asegúrate de tener los archivos descargados (`XAUUSD_D1_2000-2009_DotCom-Lehman.csv`, etc.).
2.  **Sube a Google Drive:**
    *   Ve a: `Drive > MyDrive > AchillesTraining > data`
    *   **Arrastra y suelta** todos los CSVs ahí.
    *   *Verificación:* Debes ver la lista de archivos en esa carpeta del navegador.

---

## 2. ENTRENAMIENTO EN LA NUBE ("TRAINING COLAB")
**Objetivo:** Ejecutar el notebook que entrena el cerebro (LSTM).

1.  **Sube el Notebook de Entrenamiento:**
    *   He creado el archivo: `c:\Users\David\AchillesTraining\achilles_trading_bot\Achilles_Training.ipynb`
    *   Súbelo a: `Drive > MyDrive > AchillesTraining >` (Raíz del proyecto).
2.  **Abrir con Colab:**
    *   Haz clic derecho en el archivo `.ipynb` en Drive > **Abrir con > Google Colaboratory**.
3.  **Ejecución (El Botón Play):**
    *   **Paso 1 (Mount Drive):** Ejecuta la primera celda. Te pedirá permisos para acceder a Drive. Acepta.
    *   **Paso 2 (Install):** Ejecuta la celda de dependencias.
    *   **Paso 3 (Load Data):** Ejecuta la carga. Verás mensajes como `Loading XAUUSD_...`. Si sale "Warning: Missing", revisa el paso 1.
    *   **Paso 4, 5, 6 (Train):** Ejecuta secuencialmente. Verás la barra de progreso de Keras.
4.  **Resultado Final:**
    *   Al finalizar, verás en la carpeta `data/output/v3.1/` de tu Drive:
        *   `best_model.keras`
        *   `achilles_scaler.pkl`

---
**¡Listo! Una vez tengas esos archivos, pasamos a la Fase 3 (Local Brain).**
