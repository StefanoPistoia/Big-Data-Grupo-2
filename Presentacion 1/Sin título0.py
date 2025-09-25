# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 04:21:44 2025

@author: Stefano
"""

import pyreadstat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df, meta = pyreadstat.read_sav(
    "CY08MSP_STU_QQQ.SAV"
)

df.to_parquet("CY08MSP_STU_QQQ.parquet", compression="snappy")




# Load only needed columns or rows on demand
df = pd.read_parquet("CY08MSP_STU_QQQ.parquet")

latam = [
    "ARG", "BRA", "CHL", "COL", "CRI", "DOM", "GTM", "MEX", 
    "PAN", "PER", "PRY", "SLV", "URY"
]

# Filtrar el DataFrame
df = df[df["CNT"].isin(latam)]

df.to_parquet("PISA_LATAM.parquet", compression="snappy")

df = df.reset_index()

from upsetplot import UpSet, from_indicators

# Build missingness indicators
missing_ind = from_indicators(df.isna())
UpSet(missing_ind, subset_size='count').plot()

def missing_summary(df):
    n = df.isna().sum()
    percent = (n / len(df)) * 100
    summary = pd.DataFrame({
        "variable": n.index,
        "n_missing": n.values,
        "pct_missing": percent.values.round(2)
    })
    return summary.sort_values("pct_missing", ascending=False).reset_index(drop=True)

# Example usage
summary_df = missing_summary(df)
print(summary_df.head(20))



# Heatmap 
df_completitud = (
    df.groupby("CNT")
      .apply(lambda g: g.notna().mean() * 100)  # % de no nulos
)

# --- heatmap ---
plt.figure(figsize=(14,8))
sns.heatmap(df_completitud, 
            cmap="RdBu", 
            vmin=0, vmax=100, 
            cbar_kws={'label': 'Porcentaje de datos no-nulos'})

plt.title("PISA 2022 LATAM - Porcentaje de información disponible por país y variable", fontsize=14)
plt.xlabel("Variables")
plt.ylabel("País")
plt.tight_layout()
plt.show()















import requests
import os

# Mapeo de ISO3 -> ISO2
iso_map = {
    "ARG": "AR",
    "BRA": "BR",
    "CHL": "CL",
    "COL": "CO",
    "CRI": "CR",
    "DOM": "DO",
    "GTM": "GT",
    "MEX": "MX",
    "PAN": "PA",
    "PER": "PE",
    "PRY": "PY",
    "SLV": "SV",
    "URY": "UY"
}

# Crear carpeta para guardar banderas
os.makedirs("flags", exist_ok=True)

# Descargar cada bandera
for iso3, iso2 in iso_map.items():
    url = f"https://flagcdn.com/w80/{iso2.lower()}.png"  # w80 = 80px de ancho
    path = f"flags/{iso2}.png"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"✅ Bandera guardada: {path}")
    except Exception as e:
        print(f"❌ No se pudo bajar {iso3} ({iso2}): {e}")


from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


# --- heatmap ---
plt.figure(figsize=(14,8))
ax = sns.heatmap(df_completitud, cmap="RdBu", vmin=0, vmax=100,
                 cbar_kws={'label': 'Porcentaje de datos no-nulos'})
plt.title("Porcentaje de información disponible por país y variable", fontsize=14)
plt.xlabel("Variables")
plt.ylabel("País (CNT)")

yticks = ax.get_yticks()
labels = [t.get_text() for t in ax.get_yticklabels()]

for y, iso3 in zip(yticks, labels):
    iso2 = iso_map.get(iso3, None)
    if iso2:
        flag_path = f"flags/{iso2}.png"  # carpeta donde guardes las banderas
        try:
            img = mpimg.imread(flag_path)
            imagebox = OffsetImage(img, zoom=0.05)  # ajustar tamaño
            ab = AnnotationBbox(imagebox, (-0.5, y), 
                                frameon=False, box_alignment=(1,0.5))
            ax.add_artist(ab)
        except FileNotFoundError:
            print(f"⚠️ Falta la bandera de {iso3}")

plt.tight_layout()
plt.show()


#graficos

import matplotlib.ticker as mtick

# ... (Tu código de preparación de datos, asumiendo que 'df' existe) ...

# Re-codificar variable Likert fusionando 1 y 2
df["ST273Q06JA_rec"] = df["ST273Q06JA"].replace({
    1: "Todas o la mayoría de las clases",
    2: "Todas o la mayoría de las clases",
    3: "Algunas clases",
    4: "Nunca o casi nunca"
})

# Frecuencias por país
freq = (
    df.groupby(["CNT", "ST273Q06JA_rec"])
    .size()
    .unstack(fill_value=0)
)

# Pasar a proporciones (0–1)
prop = freq.div(freq.sum(axis=1), axis=0)

# Nos quedamos solo con la categoría negativa
neg = prop["Todas o la mayoría de las clases"]

# Ordenar países de mayor a menor problema (Para gráfico vertical, es mejor ordenar
# por orden de aparición en el eje X, así que ordenaremos descendente y luego
# usaremos la serie tal cual)
neg = neg.sort_values(ascending=False) # Ordenar de mayor a menor problema (el país con
                                       # la barra más alta aparecerá primero a la izquierda)

# Graficar
fig, ax = plt.subplots(figsize=(10, 6)) # Un poco más ancho para los nombres de los países

# USAR ax.bar() para barras verticales
ax.bar(
    neg.index,    # Eje X: Países (índice)
    neg.values,   # Eje Y: Porcentajes (valores)
    color="#d73027",
    width=0.8     # 'width' achica las barras y las junta (en vertical)
)

# Eje en porcentaje - Ahora en el EJE Y
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Forzar nombres de países horizontales: ¡NECESITAS ROTAR LAS ETIQUETAS DEL EJE X!
plt.xticks(rotation=45, ha='right') # Rota 45 grados y alinea a la derecha para mejor legibilidad

# Texto claro
ax.set_xlabel("País")
ax.set_ylabel("Porcentaje de respuestas")
ax.set_title("Distracción frecuente por uso de recursos digitales en clase",
             fontsize=14, weight="bold")

# Estética
ax.spines["top"].set_visible(False)
ax.spines



import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import the colormap module

# ... (Your code for data preparation to get the 'neg' Series) ...

# For demonstration, assuming 'neg' is a pandas Series sorted descending:
# neg = neg.sort_values(ascending=False)

# Graficar
fig, ax = plt.subplots(figsize=(10, 6))

# --- COLOR MAPPING STEPS ---

# 1. Choose a colormap. 'Reds' works well here.
cmap = cm.get_cmap('Reds')

# 2. Normalize the values to the 0-1 range expected by the colormap.
# We normalize the 'neg.values' (percentages) so that the highest value 
# gets the darkest red, and the lowest gets the lightest.
norm = plt.Normalize(neg.min(), neg.max())

# 3. Get the colors for each bar by applying the normalized values to the colormap.
bar_colors = cmap(norm(neg.values))

# --- PLOTTING ---

ax.bar(
    neg.index,
    neg.values,
    color=bar_colors,  # Use the calculated array of colors
    width=0.8
)

# Eje en porcentaje - Eje Y
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Rotar las etiquetas del Eje X para legibilidad
plt.xticks(rotation=45, ha='right')

# Texto claro
ax.set_xlabel("País")
ax.set_ylabel("Porcentaje de respuestas")
ax.set_title("Distracción frecuente por uso de recursos digitales en clase",
             fontsize=14, weight="bold")

# Estética
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()


#calculo de puntaje mate por pais 
import numpy as np


# 1. Definir las listas de columnas necesarias
# Las 10 columnas de valores plausibles para matemáticas
math_pv_cols = [f'PV{i}MATH' for i in range(1, 11)]

# La columna del peso final del estudiante
main_weight_col = 'W_FSTUWT'

# Las 80 columnas de pesos de replicación
rep_weight_cols = [f'W_FSTURWT{i}' for i in range(1, 81)]


# 2. Definir la función para calcular los estadísticos por país
def calcular_estadisticas_pisa(datos_pais):
    """
    Calcula el puntaje medio y el error estándar para un país específico
    usando valores plausibles y pesos de replicación.
    """
    # Seleccionar solo las columnas necesarias para evitar errores con NaNs
    columnas_necesarias = math_pv_cols + [main_weight_col] + rep_weight_cols
    datos_pais = datos_pais[columnas_necesarias].dropna()

    if datos_pais.empty:
        return pd.Series({'Puntaje_Medio': np.nan, 'Error_Estandar': np.nan})

    # Extraer los datos y pesos
    math_pv = datos_pais[math_pv_cols]
    main_weight = datos_pais[main_weight_col]
    rep_weights = datos_pais[rep_weight_cols]

    # --- Cálculo del puntaje medio (Point Estimate) ---
    # Calcular el promedio ponderado para cada uno de los 10 valores plausibles
    pv_means = [np.average(math_pv[col], weights=main_weight) for col in math_pv_cols]
    
    # El puntaje medio final (T) es el promedio de los 10 promedios anteriores
    puntaje_medio = np.mean(pv_means)

    # --- Cálculo del Error Estándar (Standard Error) ---
    # 1. Varianza de medición (Measurement Variance, v_m)
    # Es la varianza entre los 10 puntajes medios calculados
    varianza_medicion = np.var(pv_means, ddof=1) # ddof=1 para varianza muestral

    # 2. Varianza de muestreo (Sampling Variance, v_s)
    # Para cada peso de replicación, calcular el puntaje medio de cada PV
    rep_means_list = []
    for rep_col in rep_weights:
        current_rep_means = [np.average(math_pv[col], weights=rep_weights[rep_col]) for col in math_pv_cols]
        rep_means_list.append(current_rep_means)
    
    # Calcular la suma de las diferencias al cuadrado entre las medias de replicación y las medias originales
    sum_sq_diff = sum( (np.array(rep_means_list[r]) - np.array(pv_means))**2 for r in range(80) )
    
    # La varianza de muestreo es el promedio de estas diferencias, multiplicado por un factor (0.05 para PISA)
    varianza_muestreo = 0.05 * np.mean(sum_sq_diff)
    
    # 3. Error estándar final (SE)
    # Se combinan ambas varianzas
    # SE = sqrt( Varianza_Muestreo + (1 + 1/10) * Varianza_Medición )
    error_estandar = np.sqrt(varianza_muestreo + (1 + 1/10) * varianza_medicion)

    return pd.Series({'Puntaje_Medio': puntaje_medio, 'Error_Estandar': error_estandar})


# 3. Aplicar la función a cada país en el DataFrame
# Agrupamos por la columna 'CNT' y aplicamos nuestra función
resultados_por_pais = df.groupby('CNT').apply(calcular_estadisticas_pisa)

# Reiniciar el índice para que 'CNT' vuelva a ser una columna
resultados_por_pais = resultados_por_pais.reset_index()

# Mostrar los resultados
print("Resultados de PISA 2022 por País:")
print(resultados_por_pais)



#mapa

import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Crear el DataFrame con los datos proporcionados
data = {
    'CNT': ['ARG', 'BRA', 'CHL', 'COL', 'CRI', 'DOM', 'GTM', 'MEX', 'PAN', 'PER', 'PRY', 'SLV', 'URY'],
    'Puntaje_Medio': [377.529029, 378.691367, 411.696571, 382.696835, 384.576041, 339.107247, 
                      344.199488, 395.029919, 356.573504, 391.235785, 337.539750, 343.469101, 408.712066],
    'Error_Estandar': [2.251655, 1.581419, 2.077902, 3.031013, 1.887708, 1.623352, 
                       2.210765, 2.270589, 2.840130, 2.341028, 2.157969, 2.002411, 2.023865]
}

resultados_por_pais = pd.DataFrame(data)

print("Cargando archivo custom.geo.json...")

# Cargar el archivo GeoJSON local

gdf = gpd.read_file('custom.geo.json')


# Mapeo de códigos a posibles variaciones en el GeoJSON
codigo_mapping = {
    'ARG': ['ARG', 'AR', 'Argentina', 'ARGENTINA'],
    'BRA': ['BRA', 'BR', 'Brasil', 'Brazil', 'BRASIL', 'BRAZIL'],
    'CHL': ['CHL', 'CL', 'Chile', 'CHILE'],
    'COL': ['COL', 'CO', 'Colombia', 'COLOMBIA'],
    'CRI': ['CRI', 'CR', 'Costa Rica', 'COSTA RICA'],
    'DOM': ['DOM', 'DO', 'Dominican Republic', 'República Dominicana', 'DOMINICAN REPUBLIC'],
    'GTM': ['GTM', 'GT', 'Guatemala', 'GUATEMALA'],
    'MEX': ['MEX', 'MX', 'México', 'Mexico', 'MÉXICO', 'MEXICO'],
    'PAN': ['PAN', 'PA', 'Panamá', 'Panama', 'PANAMÁ', 'PANAMA'],
    'PER': ['PER', 'PE', 'Perú', 'Peru', 'PERÚ', 'PERU'],
    'PRY': ['PRY', 'PY', 'Paraguay', 'PARAGUAY'],
    'SLV': ['SLV', 'SV', 'El Salvador', 'EL SALVADOR'],
    'URY': ['URY', 'UY', 'Uruguay', 'URUGUAY']
}

# Función para encontrar la columna de identificación correcta
def encontrar_columna_pais(gdf, codigo_mapping):
    posibles_columnas = []
    
    for col in gdf.columns:
        if col.upper() in ['ISO_A3', 'ISO3', 'CODE', 'COUNTRY_CODE', 'ADM0_A3', 'NAME', 'COUNTRY', 'ADMIN']:
            posibles_columnas.append(col)
    
    # Probar cada columna posible
    for col in posibles_columnas:
        matches = 0
        for cnt_code, variaciones in codigo_mapping.items():
            for variacion in variaciones:
                if any(str(val).upper() == variacion.upper() for val in gdf[col] if pd.notna(val)):
                    matches += 1
                    break
        
        if matches >= 5:  # Si encontramos al menos 5 países, es buena columna
            return col, matches
    
    return None, 0

# Encontrar la columna correcta
columna_pais, matches_found = encontrar_columna_pais(gdf, codigo_mapping)

if columna_pais:
    print(f"\n🎯 Columna identificada: '{columna_pais}' con {matches_found} coincidencias")
else:
    print(f"\n⚠️  No se encontró columna de países. Usando la primera columna de texto disponible")
    # Usar la primera columna que no sea geometry
    columna_pais = [col for col in gdf.columns if col != 'geometry'][0]

print(f"📋 Valores únicos en '{columna_pais}':")
print(sorted([str(v) for v in gdf[columna_pais].unique() if pd.notna(v)][:20]))

# Crear columna de código normalizado
def normalizar_codigo(valor, codigo_mapping):
    if pd.isna(valor):
        return None
    
    valor_str = str(valor).strip().upper()
    
    for cnt_code, variaciones in codigo_mapping.items():
        if valor_str in [v.upper() for v in variaciones]:
            return cnt_code
    
    return None

gdf['CNT'] = gdf[columna_pais].apply(lambda x: normalizar_codigo(x, codigo_mapping))

# Filtrar solo países de LATAM con datos
latam_gdf = gdf[gdf['CNT'].isin(resultados_por_pais['CNT'])].copy()

print(f"\n🌎 Países de LATAM encontrados: {len(latam_gdf)}")
print(f"📝 Países encontrados: {sorted(latam_gdf['CNT'].tolist())}")

# Merge con los datos de puntajes
latam_data = latam_gdf.merge(resultados_por_pais, on='CNT', how='inner')

print(f"🔗 Países con datos después del merge: {len(latam_data)}")

# Crear el mapa
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Colormap personalizado
colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4']
cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

# Plotear todos los países del GeoJSON en gris (contexto)
gdf.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)

# Plotear los países de LATAM con colores según puntaje
if len(latam_data) > 0:
    latam_data.plot(column='Puntaje_Medio', 
                    cmap=cmap, 
                    ax=ax, 
                    legend=False,
                    edgecolor='black',
                    linewidth=1.2,
                    alpha=0.9)
    
    # Agregar etiquetas
    for idx, row in latam_data.iterrows():
        centroid = row['geometry'].centroid
        ax.annotate(f"{row['CNT']}\n{row['Puntaje_Medio']:.1f}", 
                   xy=(centroid.x, centroid.y),
                   ha='center', va='center',
                   fontsize=10, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='white', 
                           alpha=0.9,
                           edgecolor='black',
                           linewidth=1))

# Configurar límites para América Latina
bounds = latam_data.total_bounds if len(latam_data) > 0 else gdf.total_bounds
margin = 5
ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

# Título y configuración
ax.set_title('PISA 2022 Puntaje promedio en Matemática por país', 
             fontsize=24, fontweight='bold', pad=25)

# Quitar ejes para un look más limpio
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Colorbar
if len(latam_data) > 0:
    min_puntaje = resultados_por_pais['Puntaje_Medio'].min()
    max_puntaje = resultados_por_pais['Puntaje_Medio'].max()
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_puntaje, vmax=max_puntaje))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30, pad=0.02)
    cbar.ax.tick_params(labelsize=12)

# Estadísticas
stats_text = f""" Estadísticas:
 Puntaje máximo: {resultados_por_pais['Puntaje_Medio'].max():.1f} (Chile)
 Puntaje mínimo: {resultados_por_pais['Puntaje_Medio'].min():.1f} (Paraguay)
 Promedio regional: {resultados_por_pais['Puntaje_Medio'].mean():.1f}
 Desviación estándar: {resultados_por_pais['Puntaje_Medio'].std():.1f}
 Países evaluados: {len(resultados_por_pais)}"""

# Posicionar el texto en una esquina
ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=18,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.show()

# Ranking detallado
print("\n" + "="*70)
print("🏆 RANKING DETALLADO DE PAÍSES POR PUNTAJE PROMEDIO")
print("="*70)

ranking = resultados_por_pais.sort_values('Puntaje_Medio', ascending=False).reset_index(drop=True)

nombres_paises = {
    'CHL': '🇨🇱 Chile', 'URY': '🇺🇾 Uruguay', 'MEX': '🇲🇽 México', 'PER': '🇵🇪 Perú',
    'CRI': '🇨🇷 Costa Rica', 'COL': '🇨🇴 Colombia', 'BRA': '🇧🇷 Brasil', 'ARG': '🇦🇷 Argentina',
    'PAN': '🇵🇦 Panamá', 'GTM': '🇬🇹 Guatemala', 'SLV': '🇸🇻 El Salvador', 
    'DOM': '🇩🇴 Rep. Dominicana', 'PRY': '🇵🇾 Paraguay'
}

for idx, row in ranking.iterrows():
    pos = idx + 1
    pais_nombre = nombres_paises.get(row['CNT'], f"🏳️ {row['CNT']}")
    puntaje = row['Puntaje_Medio']
    error = row['Error_Estandar']
    
    if pos <= 3:
        medal = '🥇' if pos == 1 else '🥈' if pos == 2 else '🥉'
        print(f"{medal} {pos:2d}. {pais_nombre:<20} │ {puntaje:6.1f} ± {error:4.1f}")
    else:
        print(f"   {pos:2d}. {pais_nombre:<20} │ {puntaje:6.1f} ± {error:4.1f}")

print("="*70)
print(f"✅ Mapa generado usando: custom.geo.json")
print(f"📍 Países mapeados exitosamente: {len(latam_data)}/{len(resultados_por_pais)}")

if len(latam_data) < len(resultados_por_pais):
    faltantes = set(resultados_por_pais['CNT']) - set(latam_data['CNT'])
    print(f"⚠️  Países no encontrados en el GeoJSON: {', '.join(sorted(faltantes))}")