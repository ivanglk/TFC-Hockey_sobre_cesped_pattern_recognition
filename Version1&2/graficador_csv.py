import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN ACADÉMICA
# ==========================================
# ⚠️ Ruta de tu carpeta principal y del archivo CSV
ruta_carpeta = r"C:\Users\ivang\Desktop\Tesis_Hockey"
ruta_csv = os.path.join(ruta_carpeta, "recuperaciones_hockey_final_v8_ Corrida de 17 40 a 19 10.csv")

df = pd.read_csv(ruta_csv)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
colores_equipos = {"Local (Vélez)": "#2ecc71", "Visita (DAOM)": "#e74c3c"}

# ==========================================
# PROCESAMIENTO DE EFICACIA POSICIONAL
# ==========================================
# Reclasificar zonas en "Mitad Defensiva" y "Mitad Ofensiva" según el equipo
def clasificar_mitad(row):
    zona = row['Zona_Recuperacion']
    equipo = row['Equipo_Recuperador']
    
    if equipo == "Local":
        if zona in ["Z1_ArcoLocal_25yd", "Z2_25yd_50yd_Local"]: return "Mitad Defensiva (Z1-Z2)"
        else: return "Mitad Ofensiva (Z3-Z4)"
    else: # Visita
        if zona in ["Z4_25yd_ArcoVisita", "Z3_50yd_25yd_Visita"]: return "Mitad Defensiva (Z4-Z3)"
        else: return "Mitad Ofensiva (Z1-Z2)"

df['Sector_Tactico'] = df.apply(clasificar_mitad, axis=1)
df['Equipo'] = df['Equipo_Recuperador'].replace({'Local': 'Local (Vélez)', 'Visita': 'Visita (DAOM)'})

# ==========================================
# GRÁFICO 1: COMPARATIVA DE EFICACIA POR MITAD DE CANCHA
# ==========================================
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x="Equipo", hue="Sector_Tactico", palette="muted", edgecolor='black')

plt.title("Comparativa de Eficacia de Recuperación por Postura Táctica", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Equipo", fontsize=12, fontweight='bold')
plt.ylabel("Volumen de Recuperaciones", fontsize=12, fontweight='bold')
plt.legend(title="Sector del Campo", fontsize=11, title_fontsize=12)

# Etiquetas numéricas
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 4), textcoords='offset points')

plt.tight_layout()

# GUARDADO SEGURO GRÁFICO 1
ruta_grafico_1 = os.path.join(ruta_carpeta, "Eficacia_Posicional.png")
plt.savefig(ruta_grafico_1, dpi=300)
print(f"✅ Gráfico 1 guardado exitosamente en: {ruta_grafico_1}")

# ==========================================
# GRÁFICO 2: MAPA DE CALOR (HEATMAP) DE ZONAS Z1 a Z4
# ==========================================
# Preparar matriz de contingencia
matriz_zonas = pd.crosstab(df['Equipo'], df['Zona_Recuperacion'])
# Ordenar columnas lógicamente de arco a arco
columnas_ordenadas = ["Z1_ArcoLocal_25yd", "Z2_25yd_50yd_Local", "Z3_50yd_25yd_Visita", "Z4_25yd_ArcoVisita"]
# Completar si falta alguna zona
for col in columnas_ordenadas:
    if col not in matriz_zonas.columns: matriz_zonas[col] = 0
matriz_zonas = matriz_zonas[columnas_ordenadas]

plt.figure(figsize=(10, 4))
sns.heatmap(matriz_zonas, annot=True, fmt="d", cmap="YlOrRd", cbar=True, 
            linewidths=.5, annot_kws={"size": 14, "weight": "bold"})

plt.title("Mapa de Calor Táctico: Volumen de Recuperaciones por Zona Específica", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Zonas del Campo (Perspectiva Longitudinal)", fontsize=12, fontweight='bold')
plt.ylabel("Equipo", fontsize=12, fontweight='bold')
plt.xticks(rotation=15)

plt.tight_layout()

# GUARDADO SEGURO GRÁFICO 2
ruta_grafico_2 = os.path.join(ruta_carpeta, "Mapa_Calor_Zonas.png")
plt.savefig(ruta_grafico_2, dpi=300)
print(f"✅ Gráfico 2 guardado exitosamente en: {ruta_grafico_2}")
print("\n¡Los gráficos están listos en tu escritorio para ser pegados en tu documento de tesis!")