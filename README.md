# 🏑 Análisis de Jugadas de Hockey sobre Césped mediante Reconocimiento de Patrones

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-orange.svg)

## 📌 Descripción del Proyecto
Este repositorio contiene el código fuente del prototipo desarrollado como Trabajo Final de Carrera (TFC) para la titulación de Ingeniero en Informática. 

El objetivo principal es automatizar y optimizar el análisis táctico en el hockey sobre césped —una tarea tradicionalmente manual, como el conteo de recuperaciones de bocha— utilizando técnicas avanzadas de Visión Artificial (Computer Vision) y Machine Learning. El sistema procesa grabaciones de partidos para extraer métricas objetivas de rendimiento.

## ✨ Características Principales
* **Detección y Seguimiento:** Identificación de jugadores y la bocha mediante modelos YOLO y algoritmos de flujo óptico.
* **Análisis de Posesión:** Cálculo automatizado de los tiempos de posesión y transiciones del equipo.
* **Métrica xG (Expected Goals):** Evaluación probabilística de las oportunidades de gol generadas durante el juego.
* **Caso de Estudio:** Las pruebas y validaciones del modelo se realizaron utilizando grabaciones reales de encuentros (ej. Vélez vs. DAOM).

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python
* **Visión Artificial:** OpenCV
* **Modelos de Detección:** [Especificar versión, ej: YOLOv8]
* **Procesamiento de Datos:** [Ej: Pandas, NumPy]

## 🚀 Instalación y Configuración

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/ivanglk/TFC-Hockey_sobre_cesped_pattern_recognition.git](https://github.com/ivanglk/TFC-Hockey_sobre_cesped_pattern_recognition.git)



2. Navega al directorio del proyecto:
   cd TFC-Hockey_sobre_cesped_pattern_recognition

3. Crea un entorno virtual e instala las dependencias:
   python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
pip install -r requirements.txt

💻 Uso
Para ejecutar el prototipo de análisis sobre un video local, utiliza el siguiente comando:
python main.py --video [ruta_al_video.mp4]

Nota Importante: 
Debido a las restricciones de tamaño de GitHub, los archivos de video originales utilizados para el entrenamiento y las pruebas (como los partidos grabados) no están incluidos en este repositorio. 
Asegúrate de colocar tus propios archivos multimedia en la carpeta data/videos/ (la cual está ignorada en el .gitignore).

👨‍💻 Autor
[Iván Glinka] Estudiante de Ingeniería en Informática
