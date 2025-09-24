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


