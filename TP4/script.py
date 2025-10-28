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
import numpy as np
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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
X = ['horastrab','educ','edad','edad2','cobertura_medica','sexo','ESTADO','estado_civil']

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

# Convertir variables categóricas en dummies
categorical_vars = ['cobertura_medica', 'sexo', 'ESTADO', 'estado_civil']
X_train_dummies = pd.get_dummies(X_train, columns=categorical_vars, drop_first=True, dtype=float)

# Statsmodels requiere que se añada una constante manualmente
X_train_const = sm.add_constant(X_train_dummies)

# 2. Instanciar y entrenar el modelo Logit
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# 3. Resumen del modelo
print(result.summary())

# Extraer resultados
coef = result.params
se = result.bse
z = result.tvalues
p = result.pvalues
conf = result.conf_int()
conf.columns = ['IC 2.5%', 'IC 97.5%']

# Crear DataFrame
tabla_resultados = pd.DataFrame({
    'Variable': coef.index,
    'Coef. (β)': coef.values,
    'exp(β)': np.exp(coef.values),
    'Error Std.': se.values,
    'z': z.values,
    'p-value': p.values,
    'IC 2.5%': conf['IC 2.5%'].values,
    'IC 97.5%': conf['IC 97.5%'].values
})

tabla_resultados = tabla_resultados.round(4)

# --- Exportar a Word ---
doc = Document()
doc.add_heading('Resultados del modelo Logit', level=1)

# Crear tabla
t = doc.add_table(rows=1, cols=len(tabla_resultados.columns))
t.style = 'Light List Accent 1'

# Cabeceras
hdr_cells = t.rows[0].cells
for i, col in enumerate(tabla_resultados.columns):
    hdr_cells[i].text = col

# Filas (con negrita si p < 0.05)
for _, row in tabla_resultados.iterrows():
    row_cells = t.add_row().cells
    for j, val in enumerate(row):
        cell = row_cells[j].paragraphs[0].add_run(str(val))
        if j == 0 or tabla_resultados.loc[_, 'p-value'] < 0.05:
            # Si es nombre de variable o p < 0.05 => negrita
            cell.bold = True

# Ajustar tamaño de fuente
for row in t.rows:
    for cell in row.cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(10)

# Guardar documento
doc.save('Resultados_Logit.docx')

import matplotlib.pyplot as plt



# 1. Crear rango de valores para 'educ'
educ_range = np.linspace(X_train_const['educ'].min(), X_train_const['educ'].max(), 100)

# 2. Crear un DataFrame con 'educ' variable y los demás fijos en su media
X_pred = X_train_const.copy()
for col in X_pred.columns:
    if col != 'educ':
        X_pred[col] = X_pred[col].mean()

# Reemplazamos 'educ' por el rango deseado
X_pred = X_pred.loc[X_pred.index.repeat(len(educ_range))].reset_index(drop=True)
X_pred['educ'] = np.tile(educ_range, len(X_train_const))

# 3. Predecir probabilidades
pred_probs = result.predict(X_pred)

# 4. Graficar relación entre educación y probabilidad predicha
plt.figure(figsize=(8, 5))
plt.plot(educ_range, pred_probs.groupby(X_pred['educ']).mean(), color='darkblue', linewidth=2)
plt.title('Probabilidad predicha de ser pobre según nivel educativo', fontsize=13)
plt.xlabel('Años de educación', fontsize=12)
plt.ylabel('Probabilidad predicha de ser pobre', fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(0, 1)
plt.show()
















# --- Clasificación con K-Nearest Neighbors (KNN) ---

# Usamos las mismas variables X e Y que en la regresión logística
train_df_knn = train_2025[[Y] + X].dropna()
test_df_knn = test_2025[[Y] + X].dropna()

y_train_knn = train_df_knn[Y]
X_train_knn = train_df_knn[X]

y_test_knn = test_df_knn[Y]
X_test_knn = test_df_knn[X]

# Convertir variables categóricas en dummies
X_train_knn_dummies = pd.get_dummies(X_train_knn, columns=categorical_vars, drop_first=True, dtype=float)
X_test_knn_dummies = pd.get_dummies(X_test_knn, columns=categorical_vars, drop_first=True, dtype=float)

# Alinear columnas para asegurar que train y test tengan las mismas 
X_train_aligned, X_test_aligned = X_train_knn_dummies.align(X_test_knn_dummies, join='inner', axis=1, fill_value=0)

# 2. Escalar las características
# KNN es sensible a la escala de los datos, por lo que estandarizamos las variables.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_aligned)
X_test_scaled = scaler.transform(X_test_aligned)

# 3. Iterar sobre los valores de K, entrenar y evaluar el modelo
k_values = [1, 5, 10]

for k in k_values:
    print(f"\n--- Resultados para K = {k} ---")
    
    # Instanciar el modelo
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenar el modelo
    knn.fit(X_train_scaled, y_train_knn)
    
    # Predecir en el conjunto de prueba
    y_pred_knn = knn.predict(X_test_scaled)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test_knn, y_pred_knn)
    print(f"Accuracy: {accuracy:.4f}")
    print("Reporte de Clasificación:")
    print(classification_report(y_test_knn, y_pred_knn, target_names=['No Pobre (0)', 'Pobre (1)']))
    
# --- Búsqueda del K Óptimo con Cross-Validation (KNN con K-CV) ---

from sklearn.model_selection import cross_val_score

print("\n--- Buscando el K óptimo para KNN con Cross-Validation (5-fold) ---")

# 1. Definir el rango de valores de K para probar
k_range = range(1, 11)
cv_scores = []

# 2. Realizar validación cruzada de 5 folds para cada valor de K
print("Calculando el accuracy promedio para cada K...")
for k in k_range:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    # cross_val_score divide los datos, entrena y evalúa automáticamente
    # Usamos X_train_scaled y y_train_knn, que son nuestros datos de entrenamiento
    scores = cross_val_score(knn_cv, X_train_scaled, y_train_knn, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# 3. Encontrar el K óptimo
optimal_k = k_range[np.argmax(cv_scores)]
max_accuracy = max(cv_scores)


# 4. Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o', linestyle='--', color='b', label='Accuracy de CV')
plt.axvline(optimal_k, color='r', linestyle='-', label=f'K Óptimo = {optimal_k}')
plt.title('Accuracy de KNN vs. Número de Vecinos (K) - Cross-Validation', fontsize=14)
plt.xlabel('Número de Vecinos (K)', fontsize=12)
plt.ylabel('Accuracy Promedio (5-Fold CV)', fontsize=12)
plt.xticks(k_range)
plt.legend()
plt.grid(True)
plt.show()

# 5. Comentario sobre el resultado
print("\n--- Comentario sobre el K Óptimo ---")
print(f"El gráfico muestra el rendimiento del modelo para diferentes valores de K. Un K bajo (como 1) puede llevar a un sobreajuste, ya que el modelo es muy sensible a puntos individuales y ruido. A medida que K aumenta, el modelo se generaliza mejor, y el accuracy tiende a subir.")
print(f"En este caso, el accuracy más alto se alcanza con K={optimal_k}. A partir de este punto, aumentar K podría hacer que el modelo sea demasiado simple (underfitting), perdiendo detalles importantes y disminuyendo su rendimiento.")
print(f"Por lo tanto, {optimal_k} es el número óptimo de vecinos cercanos para este problema, ya que representa el mejor balance entre sesgo y varianza según la validación cruzada.")

    
