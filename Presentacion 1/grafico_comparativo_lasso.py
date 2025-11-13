import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# --- Configuración ---
# Ruta al archivo Excel con los resultados de Lasso
ruta_resultados_lasso = 'resultados_lasso_pisa.xlsx'

# Nombres de las hojas para cada materia
materias = {
    'Matemática': 'Resultados_Matemática',
    'Lengua': 'Resultados_Lengua',
    'Ciencias': 'Resultados_Ciencias'
}

# Mapeo de códigos de país a nombres para las etiquetas del gráfico
nombres_paises = {
    'CNT_BRA': 'Brasil',
    'CNT_CHL': 'Chile',
    'CNT_COL': 'Colombia',
    'CNT_CRI': 'Costa Rica',
    'CNT_DOM': 'Rep. Dominicana',
    'CNT_GTM': 'Guatemala',
    'CNT_MEX': 'México',
    'CNT_PAN': 'Panamá',
    'CNT_PER': 'Perú',
    'CNT_PRY': 'Paraguay',
    'CNT_SLV': 'El Salvador',
    'CNT_URY': 'Uruguay'
}

# --- Creación de los Gráficos ---

# Crear una figura con 3 subplots apilados verticalmente
fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=True)
fig.suptitle('Coeficientes de Lasso para Países (Base: Argentina)', fontsize=24, weight='bold', y=0.98)

# --- Paso Clave: Unificar y ORDENAR el eje Y para todos los gráficos ---
# 1. Obtener todos los países con coeficientes no nulos en CUALQUIER materia
todos_los_paises = set()
for nombre_hoja in materias.values():
    df_temp = pd.read_excel(ruta_resultados_lasso, sheet_name=nombre_hoja, index_col=0)
    coefs_paises_temp = df_temp[df_temp.index.str.startswith('CNT_')]
    paises_no_cero_temp = coefs_paises_temp[coefs_paises_temp['coeficiente_lasso'] != 0].index
    todos_los_paises.update(paises_no_cero_temp)

# 2. Cargar los coeficientes de Matemática para usarlos como clave de ordenamiento
df_math = pd.read_excel(ruta_resultados_lasso, sheet_name=materias['Matemática'], index_col=0)
coefs_math = df_math[df_math.index.str.startswith('CNT_')]['coeficiente_lasso']

# 3. Crear un DataFrame para ordenar
# Usamos todos los países que aparecen en al menos un gráfico
df_orden = pd.DataFrame(index=list(todos_los_paises))
# Mapeamos los coeficientes de matemática. Los que no estén (porque son 0) se rellenan con 0.
df_orden['coef_math'] = df_orden.index.map(coefs_math).fillna(0)
# Ordenamos de mayor a menor
df_orden = df_orden.sort_values('coef_math', ascending=True)

# 4. La lista final de países para el eje Y, en el orden deseado
paises_eje_y = [nombres_paises[p] for p in df_orden.index]
# --- Fin del paso clave ---


# Iterar sobre cada materia para crear su gráfico correspondiente
for i, (nombre_materia, nombre_hoja) in enumerate(materias.items()):
    ax = axes[i]

    # 1. Cargar los datos de la hoja de Excel correspondiente
    try:
        df_lasso = pd.read_excel(ruta_resultados_lasso, sheet_name=nombre_hoja, index_col=0)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ruta_resultados_lasso}'. Asegúrate de que el archivo exista en el directorio correcto.")
        exit()
    except ValueError:
        print(f"Error: No se encontró la hoja '{nombre_hoja}' en el archivo Excel. Verifica los nombres de las hojas.")
        exit()

    # 2. Filtrar para obtener solo los coeficientes de los países (dummies) que no son cero
    coefs_paises = df_lasso[df_lasso.index.str.startswith('CNT_')]
    coefs_paises_no_cero = coefs_paises[coefs_paises['coeficiente_lasso'] != 0].copy()

    # 3. Preparar los datos usando la lista unificada de países
    # Creamos una Serie con los coeficientes, indexada por el nombre del país
    coefs_mapeados = coefs_paises_no_cero['coeficiente_lasso']
    coefs_mapeados.index = coefs_mapeados.index.map(nombres_paises)
    
    # Reindexamos la serie para que coincida con el orden de nuestro eje Y unificado.
    # Los países sin coeficiente en esta materia tendrán NaN.
    coefs_para_plot = coefs_mapeados.reindex(paises_eje_y)

    # 4. Crear el gráfico de barras horizontales
    # Asignar colores basados en si el coeficiente es positivo o negativo
    colores = ['#E65747' if x < 0 else '#642C80' if x > 0 else 'none' for x in coefs_para_plot]
    
    ax.barh(
        coefs_para_plot.index, # Usamos el índice de la serie reindexada (nombres de países ordenados)
        coefs_para_plot.values, # Usamos los valores (los NaN no se dibujarán)
        color=colores,
        height=0.6 # Ajusta este valor (default: 0.8) para cambiar el grosor
    )

    # 5. Añadir etiquetas y mejorar la estética
    ax.set_title(f'{nombre_materia}', fontsize=18, weight='bold', pad=15)
    ax.axvline(0, color='grey', linestyle='--', linewidth=1.5) # Línea en cero para referencia
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    
    # Ajustar los límites del eje X para evitar solapamientos
    if nombre_materia == 'Matemática':
        current_right = ax.get_xlim()[1]
        ax.set_xlim(left=-2, right=current_right * 1.1)
    elif nombre_materia == 'Lengua':
        current_left = ax.get_xlim()[0]
        ax.set_xlim(left=current_left, right=8.5)
    elif nombre_materia == 'Ciencias':
        current_left = ax.get_xlim()[0]
        ax.set_xlim(left=-3.3, right=9)
        
    # Poner las etiquetas del eje Y en negrita
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        
    # Añadir etiquetas de valor en cada barra
    for index, value in enumerate(coefs_para_plot):
        if pd.notna(value): # Solo añadir texto si el valor no es NaN
            # Determinar alineación y posición para que la etiqueta quede fuera de la barra
            ha = 'left' if value >= 0 else 'right'
            offset = ax.get_xlim()[1] * 0.015 # Offset para separar la etiqueta de la barra
            x_pos = value + offset if value >= 0 else value - offset
            
            ax.text(x_pos, index, f'{value:.2f}', va='center', ha=ha, fontsize=12, weight='bold', color='black')

# Configuración final para toda la figura
axes[1].set_xlabel('Valor del Coeficiente Lasso', fontsize=16, weight='bold', labelpad=15)
axes[0].set_ylabel('', fontsize=16, weight='bold', labelpad=15)
fig.tight_layout(rect=[0, 0.02, 1, 0.94]) # Ajustar para que el título principal no se solape


plt.show()

print(f"\n✅ Gráfico comparativo generado exitosamente.")