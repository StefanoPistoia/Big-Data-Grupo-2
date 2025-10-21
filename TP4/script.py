# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 09:35:07 2025 (actualizado)

@author: agostina.giovanardi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from docx import Document
import statsmodels.api as sm

# Cargar el archivo Excel en un DataFrame de pandas
df = pd.read_excel("db_nea_respuestas.xlsx")

# Nombre de la columna que contiene el año
columna_anio = 'ANO4'
# Semilla para la reproducibilidad
random_seed = 444

# --- Procesamiento para el año 2005 ---
respondieron_2005 = df[df[columna_anio] == 2005]

train_2005, test_2005 = train_test_split(
    respondieron_2005,
    test_size=0.3,
    random_state=random_seed
)

# --- Procesamiento para el año 2025 ---

respondieron_2025 = df[df[columna_anio] == 2025]

train_2025, test_2025 = train_test_split(
    respondieron_2025,
    test_size=0.3,
    random_state=random_seed
)

# Definimos variables
Y = 'pobre'
X = ['horastrab','educ','edad','edad2','adulto_equiv']

# --- Tabla de diferencia de medias con p-values entre train y test para cada año ---

results = []

# Diccionario para mapear el año a sus dataframes de train y test
dataframes_por_anio = {
    2005: {'train': train_2005, 'test': test_2005},
    2025: {'train': train_2025, 'test': test_2025}
}

print("\nCalculando diferencias de medias y p-values...")

for anio, dfs in dataframes_por_anio.items():
    train_df = dfs['train']
    test_df = dfs['test']
    
    print(f"  Procesando año: {anio}")
    
    for var in X:
        # Asegurarse de que la variable existe en ambos dataframes
        if var in train_df.columns and var in test_df.columns:
            # Calcular medias
            mean_train = train_df[var].mean()
            mean_test = test_df[var].mean()
            
            # Calcular diferencia de medias
            diff_mean = mean_train - mean_test
            
            # Realizar t-test de Student para muestras independientes
            # Usamos .dropna() para manejar posibles valores faltantes
            # equal_var=False para Welch's t-test, que no asume varianzas iguales
            t_stat, p_value = stats.ttest_ind(
                train_df[var].dropna(), 
                test_df[var].dropna(), 
                equal_var=False
            )
            
            results.append({
                'Año': anio,
                'Variable': var,
                'Media_Train': mean_train,
                'Media_Test': mean_test,
                'Diferencia_Medias': diff_mean,
                'P_Value': p_value
            })
        else:
            print(f"    Advertencia: La variable '{var}' no se encontró en los dataframes del año {anio}. Se omitirá.")

# Crear un DataFrame con los resultados
df_medias_pvalues = pd.DataFrame(results)

# Mostrar el DataFrame resultante
print("\nTabla de Diferencia de Medias y P-values:")
print(df_medias_pvalues)

# Exportar a Excel
df_medias_pvalues.to_excel("diferencia_medias_train_test.xlsx", index=False)





#fijarse que categorías son base para las dummies





# --- Regresión logística ---
train_2025_full = train_2025[[Y] + X].dropna()
# 1. Preparar los datos de entrenamiento para el modelo
y_train = train_2025_full[Y]
X_train = train_2025_full[X]

# Statsmodels requiere que se añada una constante (intercepto) manualmente
X_train_const = sm.add_constant(X_train)

# 2. Instanciar y entrenar el modelo Logit
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# 3. Resumen del modelo
print(result.summary())

import numpy as np
import pandas as pd
from docx import Document
import statsmodels.api as sm

# Ajustás tu modelo
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Extraer resumen de resultados
coef = result.params
se = result.bse
z = result.tvalues
p = result.pvalues
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']

# Crear DataFrame con todo
tabla_resultados = pd.DataFrame({
    'Coef. (β)': coef,
    'Error Std.': se,
    'z': z,
    'p-value': p,
    'IC 2.5%': conf['2.5%'],
    'IC 97.5%': conf['97.5%'],
})

# Añadir columna con Odds Ratio
tabla_resultados['exp(β)'] = np.exp(tabla_resultados['Coef. (β)'])

# Reordenar columnas
tabla_resultados = tabla_resultados[['Coef. (β)', 'exp(β)', 'Error Std.', 'z', 'p-value', 'IC 2.5%', 'IC 97.5%']]

# Redondear para prolijidad
tabla_resultados = tabla_resultados.round(4)

# --- Exportar a Word ---
doc = Document()
doc.add_heading('Resultados del modelo Logit', level=1)

# Añadir tabla al documento
t = doc.add_table(rows=1, cols=len(tabla_resultados.columns))
t.style = 'Light List Accent 1'

# Cabeceras
hdr_cells = t.rows[0].cells
for i, col in enumerate(tabla_resultados.columns):
    hdr_cells[i].text = col

# Filas
for i, row in tabla_resultados.iterrows():
    row_cells = t.add_row().cells
    for j, val in enumerate(row):
        row_cells[j].text = str(val)

# Guardar documento
doc.save('Resultados_Logit.docx')

print("✅ Tabla exportada a 'Resultados_Logit.docx'")



#añadir una columna que muestre los logs odds