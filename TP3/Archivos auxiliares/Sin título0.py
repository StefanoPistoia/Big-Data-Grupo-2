# -*- coding: utf-8 -*-
import numpy as np
import pyreadstat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

db = pd.read_excel("db_nea_respuestas.xlsx")
hogar2005 = pd.read_stata("Hogar_t105.dta")
hogar2025 = pd.read_excel("usu_hogar_T125.xlsx")

hogar2005 = hogar2005[['CODUSU', 'IX_Tot']]
hogar2005 = hogar2005.rename(columns={'IX_Tot': 'IX_TOT'})
hogar2025 = hogar2025[['CODUSU', 'IX_TOT']]
hogar = pd.concat([hogar2005, hogar2025], ignore_index=True)
hogar = hogar.drop_duplicates(subset=['CODUSU'], keep='last')
#conservamos la ultima observacion vigente



db = db.merge(hogar, on='CODUSU', how='left')


# Seleccionar las variables indicadas
vars_interes = ['edad', 'edad2', 'educ', 'ingreso_total_familiar', 'IX_TOT', 'horastrab']

# Calcular la matriz de correlaciones
corr_matrix = db[vars_interes].corr(method='pearson')

# Mostrar la matriz
print(corr_matrix)


# Visualizar con un heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Matriz de correlaciones - Región NEA')
plt.xticks(rotation=45)
plt.show()


#PCA

# Inicializamos el transformador, 
scaler = StandardScaler(with_std=True, with_mean=True) 
# Aplicamos fit_transform al DataFrame
vars_standard = pd.DataFrame(scaler.fit_transform(db[vars_interes]),columns= vars_interes)
vars_standard = vars_standard.dropna()



pca = PCA()
eph_pca = pca.fit_transform(vars_standard)
eph_pca




loading_vectors = pca.components_ # cada fila corresponde a un CP y cada columna, a una variable
print("Loadings:\n", pca.components_)
print("Loadings del CP1:\n",pca.components_[0]) 



#grafico de dispersion

i, j = 0, 1 # Componentes
fig, (ax_scores, ax_ponderadores) = plt.subplots(1, 2, figsize=(10, 5)) # 1 fila, 2 columnas

# ---
## Panel A. Score 1 y 2
ax_scores.scatter(eph_pca[:,0], eph_pca[:,1], s=30, alpha=0.7) # graficamos los valores de los CP1 y CP2
ax_scores.set_xlabel('Componente Principal %d' % (i+1))
ax_scores.set_ylabel('Componente Principal %d' % (j+1))
ax_scores.set_title('A. Score 1 y 2')

# Líneas punteadas en los ejes
ax_scores.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Eje horizontal en y=0
ax_scores.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Eje vertical en x=0

# ---
## Panel B. Ponderadores
# Líneas punteadas en los ejes
ax_ponderadores.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Eje horizontal en y=0
ax_ponderadores.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Eje vertical en x=0

# ponderadores
for k in range(pca.components_.shape[1]): # loop que itera por la cantidad de features
    ax_ponderadores.arrow(0, 0, pca.components_[i,k], pca.components_[j,k], color ='red', head_width=0.03) # flecha desde el origen (0) a las coordenadas
    ax_ponderadores.text(pca.components_[i,k], pca.components_[j,k], vars_standard.columns[k], color ='red') # al final de cada flecha, nombre de la variable

ax_ponderadores.set_xlabel('Ponderador de CP %d' % (i+1))
ax_ponderadores.set_ylabel('Ponderador de CP %d' % (j+1))
ax_ponderadores.set_title('B. Ponderadores')
ax_ponderadores.set_xlim(-1.1, 1.1)
ax_ponderadores.set_ylim(-1.1, 1.1)

plt.tight_layout() # Ajusta los subplots para que no se superpongan
plt.show()





print(pca.explained_variance_ratio_)
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) # 2 subplots uno al lado del otro
ticks = np.arange(pca.n_components_)+1 # para crear ticks en el eje horizontal
ax = axes[0]
ax.plot(ticks, pca.explained_variance_ratio_ , marker='o')
ax.set_xlabel('Nro. de Componente Principal ($M$)');
ax.set_ylabel('Prop. de la varianza explicada')
ax.set_ylim([0,1])
ax.set_xticks(ticks)
ax = axes[1]
ax.plot(ticks, pca.explained_variance_ratio_.cumsum(), marker='o') 
ax.set_xlabel('Nro. de Componente Principal ($M$)')
ax.set_ylabel('Suma acumulada de la varianza explicada')
ax.set_ylim([0, 1])
ax.set_xticks(ticks)
fig






from sklearn.cluster import KMeans, AgglomerativeClustering

vars_cluster = db[vars_interes]
vars_cluster = vars_cluster.dropna()

kmeans2 = KMeans(n_clusters=2, random_state=5, init="random", n_init=20).fit(vars_cluster)
kmeans4 = KMeans(n_clusters=4, random_state=5, init="random", n_init=20).fit(vars_cluster)
kmeans10 = KMeans(n_clusters=10, random_state=5, init="random", n_init=20).fit(vars_cluster)



# Crear columnas con los clusters
vars_cluster['cluster_k2'] = kmeans2.labels_
vars_cluster['cluster_k4'] = kmeans4.labels_
vars_cluster['cluster_k10'] = kmeans10.labels_

# Variables para el gráfico
x = 'edad'
y = 'ingreso_total_familiar'

# Crear figura con tres subgráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Lista con los modelos y sus nombres
clusters = [
    ('K=2', 'cluster_k2'),
    ('K=4', 'cluster_k4'),
    ('K=10', 'cluster_k10')
]

# Generar los gráficos
for ax, (title, col) in zip(axes, clusters):
    scatter = ax.scatter(
        vars_cluster[x],
        vars_cluster[y],
        c=vars_cluster[col],
        cmap='tab10',
        s=30,
        alpha=0.7
    )
    ax.set_title(f'K-Means con {title}')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Ingreso familiar')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()





# Posibles valores del número de cluster
n_cluster_range = range(1,11)

# Objeto para guardad los valores de inertia
inertia_values = []

# Loop para probar distintos valores de cluser
for n_clusters in n_cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=5)
    kmeans.fit(vars_cluster)
    inertia_values.append(kmeans.inertia_)
    
    
 # Ahora graficamos como cambia inertia con los distintos clusters
plt.plot(n_cluster_range,inertia_values, marker='o')
plt.xlabel('Numero de clústers')
plt.ylabel('Metrica de Inertia')
plt.title('Figura 3. Método Elbow\npara estos datos con k=2')
plt.show   























# --- 1. Preparación de los Datos ---
# Seleccionamos las variables de interés y eliminamos filas con valores nulos.
# Es crucial hacer esto al principio.
vars_cluster_original = db[vars_interes].dropna()

# Creamos una copia del DataFrame original 'db' pero solo con las filas válidas (sin nulos).
# Así, mantenemos todas las columnas originales (como 'pobre') para el análisis posterior.
db_clean = db.loc[vars_cluster_original.index].copy()

#Estandarización de las Variables

scaler = StandardScaler(with_std=True, with_mean=True)
X_scaled = scaler.fit_transform(vars_cluster_original)


# Método del Codo (Elbow Method) para encontrar K óptimo 

n_cluster_range = range(1, 11)
inertia_values = []

for n_clusters in n_cluster_range:
    # Usamos n_init='auto' para usar la mejor configuración moderna de scikit-learn
    kmeans = KMeans(n_clusters=n_clusters, random_state=5, n_init=20)
    kmeans.fit(X_scaled) # <-- ¡Importante! Usamos los datos escalados
    inertia_values.append(kmeans.inertia_)

# Graficamos el método del codo
plt.figure(figsize=(10, 6)) # Un poco más grande para mejor visualización
plt.plot(n_cluster_range, inertia_values, marker='o', linestyle='--')
plt.xlabel('Número de clústers (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para k óptimo')
plt.grid(True)
plt.show()


# --- 4. Entrenamiento de los modelos K-Means específicos ---
# Entrenamos los modelos para k=2, 4, y 10 usando los datos escalados.
kmeans2 = KMeans(n_clusters=2, random_state=5, n_init=20).fit(X_scaled)
kmeans4 = KMeans(n_clusters=4, random_state=5, n_init=20).fit(X_scaled)
kmeans10 = KMeans(n_clusters=10, random_state=5, n_init=20).fit(X_scaled)

# --- 5. Asignación de los clusters al DataFrame limpio ---
# Añadimos las etiquetas de los clusters a nuestro DataFrame 'db_clean'.
# Esto nos permite tener los clusters junto a las variables originales no escaladas.
db_clean['cluster_k2'] = kmeans2.labels_
db_clean['cluster_k4'] = kmeans4.labels_
db_clean['cluster_k10'] = kmeans10.labels_


# --- 6. Visualización de los Clusters ---
# Variables para el gráfico (usamos los nombres de las columnas)
x_var = 'edad'
y_var = 'ingreso_total_familiar'

# Crear figura con tres subgráficos
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Lista con los modelos y sus nombres para iterar
clusters_to_plot = [
    ('K=2', 'cluster_k2'),
    ('K=4', 'cluster_k4'),
    ('K=10', 'cluster_k10')
]

# Generar los gráficos iterando
for ax, (title, col) in zip(axes, clusters_to_plot):
    # Graficamos usando los datos de 'db_clean', que tienen los valores originales.
    scatter = ax.scatter(
        db_clean[x_var],
        db_clean[y_var],
        c=db_clean[col], # El color lo da la columna del cluster
        cmap='viridis',  # Un mapa de colores diferente y muy claro
        s=40,
        alpha=0.8
    )
    ax.set_title(f'Clusters con {title}')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Ingreso Total Familiar')
    ax.grid(True, linestyle='--', alpha=0.5)

# Añadimos una leyenda general a la figura
legend1 = fig.legend(*scatter.legend_elements(), title="Clusters", loc='upper right')
fig.add_artist(legend1)

plt.tight_layout(rect=[0, 0, 0.9, 1]) # Ajustamos el layout para que la leyenda no se superponga
plt.show()









# Importar la librería seaborn, que es ideal para esto
import seaborn as sns
import matplotlib.pyplot as plt

# --- Variables para el gráfico ---
x_var = 'edad'
y_var = 'ingreso_total_familiar'
cluster_col = 'cluster_k2'
dummy_col = 'pobre'

# --- Crear el gráfico ---
plt.figure(figsize=(12, 8))

# Usamos la función scatterplot de seaborn, que es muy potente
# - hue: asigna un color a cada valor de 'cluster_k2'
# - style: asigna un símbolo a cada valor de 'pobre'
sns.scatterplot(
    data=db_clean,
    x=x_var,
    y=y_var,
    hue=cluster_col,
    style=dummy_col,
    palette='viridis', # Mapa de colores para los clusters
    s=80,              # Tamaño de los puntos
    alpha=0.9
)

# --- Mejorar el diseño del gráfico ---
plt.title('Clusters (k=2) y Condición de Pobreza', fontsize=16)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Ingreso Total Familiar', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Leyenda', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para que la leyenda no se corte

plt.show()




optimal_k = 3
kmeans_k3 = KMeans(n_clusters=optimal_k, random_state=5, n_init=20).fit(X_scaled)
db_clean['cluster_k3'] = kmeans_k3.labels_

x_var = 'edad'
y_var = 'ingreso_total_familiar'
cluster_col = 'cluster_k3'
dummy_col = 'pobre'

# --- Crear el gráfico ---
plt.figure(figsize=(12, 8))

# Usamos la función scatterplot de seaborn, que es muy potente
# - hue: asigna un color a cada valor de 'cluster_k2'
# - style: asigna un símbolo a cada valor de 'pobre'
sns.scatterplot(
    data=db_clean,
    x=x_var,
    y=y_var,
    hue=cluster_col,
    style=dummy_col,
    palette='viridis', # Mapa de colores para los clusters
    s=80,              # Tamaño de los puntos
    alpha=0.9
)

# --- Mejorar el diseño del gráfico ---
plt.title('Clusters (k=3) y Condición de Pobreza', fontsize=16)
plt.xlabel('Edad', fontsize=12)
plt.ylabel('Ingreso Total Familiar', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Leyenda', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para que la leyenda no se corte

plt.show()





from scipy.cluster.hierarchy import dendrogram, linkage


# --- 2. Calcular la Matriz de Enlace (Linkage Matrix) ---
# Aquí es donde se realiza el clustering jerárquico.
# Usamos el método 'ward' sobre nuestros datos escalados.
Z = linkage(X_scaled, method= "single")


# --- 3. Generar el Dendrograma ---
# Un dendrograma es la visualización de la jerarquía de clusters.
plt.figure(figsize=(15, 7))
plt.title('Dendrograma Truncado (Últimas 5 fusiones)', fontsize=16)
plt.xlabel('Tamaño del Cluster (entre paréntesis)', fontsize=12)
plt.ylabel('Distancia', fontsize=12)

dendrogram(
    Z,
    truncate_mode='lastp',  # Muestra solo las últimas 'p' fusiones
    p=5,                   # El número de fusiones a mostrar
    show_leaf_counts=True,  # Muestra cuántos puntos originales hay en cada hoja
    show_contracted=True,   # Representa las ramas truncadas
)
plt.show()

