# 🎥 Video Extractor Project

Este repositorio contiene un conjunto de herramientas para la extracción, análisis y detección de objetos rectangulares en videos. 
Los módulos están ubicados en `src/video_extractor/`, cada uno con una funcionalidad específica.

---
## 📚 Referencias

Este proyecto toma como base parte del trabajo publicado en el siguiente repositorio:

- 🔗 [Detector de rectángulos mediante Ridge - Proyecto TFM de Neathle](https://github.com/Neathle/TFM)

Agradecimientos al autor original por compartir su investigación y código públicamente.
## 🚀 Guía de uso

### 🔧 1. Abrir y configurar los módulos

Antes de ejecutar cualquier módulo, **es importante abrir cada uno de los scripts** (` FLD, Airline`, `Roi`, `Yolo`) y ajustar manualmente los **parámetros según 
la escena**:
- Modifica rutas de entrada (videos o imágenes)
- Ajusta valores de umbral, ROI, tamaños o modelos según necesidad
- Verifica el sistema operativo y versión de Python (especialmente para Airline)

### 🖼 2. Evaluación por imágenes

Si deseas evaluar a nivel de imágenes (en vez de video), **asegúrate de cambiar las rutas de entrada** para que apunten a la carpeta con las imágenes.

---

## 🗂 Estructura del Proyecto
├── build/
├── install/
├── log/
└── src/
└── video_extractor/
├── Ridge__FLD
├── Ridge_Airline_FLD/
├── Ridge_Roi/
└── Ridge_Yolo/

IMPORTANTE: El módulo CRG está compilado para Python 3.11 en Windows, Linux y macOS. Por favor, sigue el tutorial de pybind11 para compilarlo.
El archivo .cpp se encuentra en: CRG/extractC.

🧪 DATASET SINTÉTICO – ISAAC SIM

Este proyecto fue evaluado utilizando un conjunto de datos generados de forma sintética en NVIDIA Isaac Sim, complementados con datos reales obtenidos en el entorno Robot@Home2.

Los siguientes enlaces incluyen los resultados de cada módulo (Airline, FLD, ROI, YOLO) junto con sus respectivos Ground Truths, empleados para la evaluación y comparación del desempeño.

Cada carpeta contiene:

Imágenes originales (images)

Etiquetas y anotaciones (labels)

Imágenes con el Ground Truth (outputs)

📥 Enlace de descarga del dataset sintético (Isaac Sim):
🔗 https://drive.google.com/drive/folders/1gZvHERa0tMA8oM3BWoNmzE76GrPac45u?usp=drive_link

🤖 DATASET REAL – ROBOT@HOME2

De manera complementaria, se utilizó el dataset real Robot@Home2, organizado bajo la misma estructura que el dataset sintético con el fin de mantener consistencia en los procesos de evaluación.

Cada carpeta incluye:

Imágenes originales (images)

Etiquetas y anotaciones (labels)

Imágenes con el Ground Truth (outputs)

📥 Enlace de descarga del dataset Robot@Home2:
🔗 https://drive.google.com/drive/folders/1dpP8Z00DASQUeKYnOC32W2xYegjifNwW?usp=sharing

---
📂 SessionAIRLINE_Sintético
📂 SessionAIRLINE_Robot@Home2
📂 SessionFLD_Sintético
📂 SessionFLD_Robot@Home2
📂 SessionROI_Sintético
📂 SessionROI_Robot@Home2
📂 SessionROI2_Robot@Home2
📂 SessionYOLO_Sintético
📂 SessionYOLO_Robot@Home2


