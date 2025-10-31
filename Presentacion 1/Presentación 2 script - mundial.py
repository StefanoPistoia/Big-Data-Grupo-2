import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("CY08MSP_STU_QQQ.parquet")




#%% Calculo de puntaje de matemática por estudiante

# 1. Definir las columnas de valores plausibles para matemáticas
math_pv_cols = [f'PV{i}MATH' for i in range(1, 11)]

# 2. Calcular el puntaje promedio por estudiante
# El puntaje de cada estudiante es el promedio de sus 10 valores plausibles.
df['puntaje_matematica'] = df[math_pv_cols].mean(axis=1)

# 3. Mostrar las primeras filas con la nueva columna
print("Primeas 5 filas con el puntaje de matemática por estudiante:")
print(df[['CNT', 'puntaje_matematica']].head())


#%% Calculo de puntaje de LENGUA por estudiante

# 1. Definir las columnas de valores plausibles para lectura
read_pv_cols = [f'PV{i}READ' for i in range(1, 11)]

# 2. Calcular el puntaje promedio por estudiante
df['puntaje_lengua'] = df[read_pv_cols].mean(axis=1)

# 3. Mostrar las primeras filas con la nueva columna
print("\nPrimeras 5 filas con el puntaje de lengua por estudiante:")
print(df[['CNT', 'puntaje_lengua']].head())

#%% Calculo de puntaje de CIENCIAS por estudiante

# 1. Definir las columnas de valores plausibles para ciencias
scie_pv_cols = [f'PV{i}SCIE' for i in range(1, 11)]

# 2. Calcular el puntaje promedio por estudiante
df['puntaje_ciencias'] = df[scie_pv_cols].mean(axis=1)

# 3. Mostrar las primeras filas con la nueva columna
print("\nPrimeras 5 filas con el puntaje de ciencias por estudiante:")
print(df[['CNT', 'puntaje_ciencias', 'puntaje_lengua', 'puntaje_matematica']].head())

#%% Modelo de Regresión OLS con errores clusterizados

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Calcular el porcentaje de valores faltantes para cada columna
missing_pct = df.isna().sum() / len(df) * 100

# Identificar las columnas con más del 60% de valores faltantes
columnas_a_eliminar = missing_pct[missing_pct > 30].index.tolist()

print("\n" + "="*80)
print("⚠️  Variables eliminadas del modelo por tener más de 50% de datos faltantes:")
if columnas_a_eliminar:
    for col in sorted(columnas_a_eliminar):
        print(f"- {col} ({missing_pct[col]:.2f}%)")
else:
    print("Ninguna variable superó el umbral del 60% de datos faltantes.")
print("="*80)

# Lista de variables predictoras que seleccionaste
predictores = [
    'ST004D01T','AGE', 'GRADE', 'ISCEDP', 'IMMIG', 'COBN_S', 'COBN_M', 'COBN_F', 'LANGN', 
    'REPEAT', 'MISSSC', 'SKIPPING', 'TARDYSD', 'EXERPRAC', 'STUDYHMW', 
    'WORKPAY', 'WORKHOME', 'EXPECEDU', 'MATHPREF', 'MATHEASE', 'MATHMOT', 
    'DURECEC', 'BSMJ', 'SISCO', 'RELATST', 'BELONG', 'BULLIED', 'FEELSAFE', 
    'SCHRISK', 'PERSEVAGR', 'CURIOAGR', 'COOPAGR', 'EMPATAGR', 'ASSERAGR', 
    'STRESAGR', 'EMOCOAGR', 'GROSAGR', 'INFOSEEK', 'FAMSUP', 'DISCLIM', 
    'TEACHSUP', 'COGACRCO', 'COGACMCO', 'EXPOFA', 'EXPO21ST', 'MATHEFF', 
    'MATHEF21', 'FAMCON', 'ANXMAT', 'MATHPERS', 'CREATEFF', 'CREATSCH', 
    'CREATFAM', 'CREATAS', 'CREATOOS', 'CREATOP', 'OPENART', 'IMAGINE', 
    'SCHSUST', 'LEARRES', 'PROBSELF', 'FAMSUPSL', 'FEELLAH', 'SDLEFF', 
    'MISCED', 'FISCED', 'HISCED', 'PAREDINT', 'BMMJ1', 'BFMJ2', 'HISEI', 
    'ICTRES', 'HOMEPOS', 'ESCS', 'FCFMLRTY', 'FLSCHOOL', 'FLMULTSB', 
    'FLFAMILY', 'ACCESSFP', 'FLCONFIN', 'FLCONICT', 'ACCESSFA', 'ATTCONFM', 
    'FRINFLFM', 'ICTSCH', 'ICTAVSCH', 'ICTHOME', 'ICTAVHOM', 'ICTQUAL', 
    'ICTSUBJ', 'ICTENQ', 'ICTFEED', 'ICTOUT', 'ICTWKDY', 'ICTWKEND', 'ICTREG', 
    'ICTINFO', 'ICTDISTR', 'ICTEFFIC', 'STUBMI', 'BODYIMA', 'SOCONPA', 
    'LIFESAT', 'PSYCHSYM', 'SOCCON', 'EXPWB', 'CURSUPP', 'PQMIMP', 'PQMCAR', 
    'PARINVOL', 'PQSCHOOL', 'PASCHPOL', 'ATTIMMP', 'PAREXPT', 'CREATHME', 
    'CREATACT', 'CREATOPN', 'CREATOR','CNT'
] # 'CNT' se elimina de esta lista para ser manejada por separado.
# Lista de variables a excluir explícitamente
afuera = ['COBN_S', 'COBN_M', 'COBN_F', 'LANGN']

# Lista de variables a forzar como dummies
predictores_dummies = ['TARDYSD', 'IMMIG', 'ST004D01T'] # 'CNT' se manejará por separado

# Filtrar la lista de predictores para excluir las columnas con muchos faltantes
predictores_filtrados = [p for p in predictores if p not in columnas_a_eliminar and p not in afuera and p != 'CNT']
# Separar las columnas numéricas de las que serán dummies
# Definir las materias para el bucle
materias = {
    'Matemática': 'puntaje_matematica',
    'Lengua': 'puntaje_lengua',
    'Ciencias': 'puntaje_ciencias'
}

def prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y):
    """Prepara el DataFrame para un modelo de regresión manejando tipos y dummies."""
    # 1. Crear un nuevo DataFrame con solo las columnas necesarias
    columnas_necesarias = predictores_filtrados + [variable_y, 'CNT']
    df_modelo_temp = df[columnas_necesarias].copy()

    # 2. Convertir columnas a numérico y separar para dummificación
    numeric_cols_for_model = []
    categorical_cols_to_dummify = []
    for col in predictores_filtrados:
        df_modelo_temp[col] = pd.to_numeric(df_modelo_temp[col], errors='coerce')
        if col in predictores_dummies:
            categorical_cols_to_dummify.append(col)
        else:
            numeric_cols_for_model.append(col)

    # 3. Crear variables dummy
    df_dummies_country = pd.get_dummies(df_modelo_temp[['CNT']], columns=['CNT'], drop_first=True, dummy_na=False).astype(int)
    df_dummies_other = pd.get_dummies(df_modelo_temp[categorical_cols_to_dummify].astype('Int64'), columns=categorical_cols_to_dummify, drop_first=True, dummy_na=False).astype(int)

    # 4. Unir los DataFrames y manejar NaNs
    df_modelo_final = pd.concat([
        df_modelo_temp[numeric_cols_for_model],
        df_dummies_country,
        df_dummies_other,
        df_modelo_temp[[variable_y]]
    ], axis=1)
    df_modelo_final.dropna(inplace=True)
    
    predictores_finales = numeric_cols_for_model + list(df_dummies_country.columns) + list(df_dummies_other.columns)
    
    return df_modelo_final, predictores_finales

# Definir las materias para el bucle
materias = {
    'Matemática': 'puntaje_matematica',
    'Lengua': 'puntaje_lengua',
    'Ciencias': 'puntaje_ciencias'
}

# Crear un ExcelWriter para guardar los resultados
output_excel_path = 'resultados_regresion_pisa.xlsx'
writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f"📊 EJECUTANDO MODELO DE REGRESIÓN OLS PARA: {nombre_materia.upper()}")
    print("="*80)

    # 2. Preparar los datos usando la función refactorizada
    df_modelo_final, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)

    print(f"\nSe usarán {len(df_modelo_final)} observaciones completas para el modelo de regresión después de manejar tipos de datos.")

    # Definir X e y
    y = df_modelo_final[variable_y]
    X = df_modelo_final[predictores_finales]

    # 3. Dividir los datos en conjuntos de entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Tamaño del conjunto de prueba: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Agregar una constante (intercepto) a los conjuntos de entrenamiento y prueba
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # 4. Ajustar el modelo OLS en el conjunto de entrenamiento
    # Usamos errores robustos (HC1) ya que controlamos por país con variables dummy.
    modelo_ols = sm.OLS(y_train, X_train)
    resultados = modelo_ols.fit(cov_type='HC1')

    # 5. Realizar predicciones en el conjunto de prueba y evaluar el modelo
    y_pred = resultados.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 6. Mostrar un resumen simple en la consola y guardar el completo en Excel
    print("\n--- Resumen del Modelo ---")
    print(f"Variable Dependiente: {resultados.model.endog_names}")
    print(f"R-cuadrado ajustado: {resultados.rsquared_adj:.4f}")
    print(f"Observaciones: {int(resultados.nobs)}")
    
    print("\n--- Evaluación en el Conjunto de Prueba ---")
    print(f"R-cuadrado (R²): {r2:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")

    # Intentar imprimir el resumen completo, pero si es muy grande, solo mostrar un aviso.
    try:
        print(resultados.summary())
    except AssertionError:
        print("\n⚠️  El resumen del modelo es demasiado grande para mostrarlo completo en la consola.")
        print("    Los resultados detallados se guardarán en el archivo Excel.")
    print("="*80 + "\n")
   
    # --- Guardar resultados en Excel ---
    # Construir el DataFrame de resultados directamente para evitar errores de formato
    resumen_df = pd.DataFrame({
        'coef': resultados.params,
        'std err': resultados.bse,
     't': resultados.tvalues,
        'P>|t|': resultados.pvalues,
        '[0.025': resultados.conf_int()[0],
        '0.975]': resultados.conf_int()[1]
    })
    
    # Escribir el DataFrame en una hoja de Excel específica para la materia
    resumen_df.to_excel(writer, sheet_name=f'Resultados_{nombre_materia}')
    print(f"✅ Resultados para {nombre_materia} guardados en la hoja '{nombre_materia}' del archivo '{output_excel_path}'")

# Guardar y cerrar el archivo de Excel
writer.close()
print(f"\n🎉 ¡Análisis completado! Todos los resultados han sido guardados en '{output_excel_path}'")

#%% Modelo Random Forest con Fine-Tuning y Validación Cruzada

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f"🌳 EJECUTANDO RANDOM FOREST CON HYPERPARAMETER TUNING PARA: {nombre_materia.upper()}")
    print("="*80)

    # 1. Preparar los datos usando la función refactorizada
    df_modelo_final, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)

    print(f"\nSe usarán {len(df_modelo_final)} observaciones completas para el modelo.")

    y = df_modelo_final[variable_y]
    X = df_modelo_final[predictores_finales]

    # 2. División de datos en Entrenamiento+Validación (80%) y Prueba (20%)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Tamaño del conjunto de Desarrollo (Entrenamiento + Validación): {len(X_dev)} ({len(X_dev)/len(X)*100:.1f}%)")
    print(f"Tamaño del conjunto de Prueba Final: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # 3. Definir el espacio de hiperparámetros para RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', 1.0],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True] # OOB score solo está disponible si bootstrap=True
    }

    # 4. Configurar y ejecutar la búsqueda aleatoria con validación cruzada (k-validation)
    rf = RandomForestRegressor(random_state=42, oob_score=True)
    
    # n_iter controla cuántas combinaciones de parámetros se prueban.
    # cv=3 significa 3-fold cross-validation.
    # n_jobs=-1 usa todos los procesadores disponibles.
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=10, # Probar 10 combinaciones. Aumentar para una búsqueda más exhaustiva.
        cv=5, 
        verbose=2, 
        random_state=42, 
        n_jobs=-1,
        scoring='neg_root_mean_squared_error' # Métrica para optimizar
    )

    print("\nIniciando búsqueda de hiperparámetros con RandomizedSearchCV...")
    random_search.fit(X_dev, y_dev)

    print("\n--- Mejores Hiperparámetros Encontrados ---")
    print(random_search.best_params_)

    # 5. Evaluar el mejor modelo en el conjunto de prueba final
    best_rf = random_search.best_estimator_
    y_pred_final = best_rf.predict(X_test)

    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
    r2_final = r2_score(y_test, y_pred_final)

    print("\n--- Resultados de la Evaluación Final en el Conjunto de Prueba (20%) ---")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_final:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2_final:.4f}")
    print(f"Out-of-Bag (OOB) Score del mejor modelo (entrenado en 80%): {best_rf.oob_score_:.4f}")
    print("="*80 + "\n")

#%% Modelo Lasso con Cross-Validation

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Crear un nuevo ExcelWriter para los resultados de Lasso
output_excel_path_lasso = 'resultados_lasso_pisa.xlsx'
writer_lasso = pd.ExcelWriter(output_excel_path_lasso, engine='xlsxwriter')

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f" LASSO REGRESSION CON CROSS-VALIDATION PARA: {nombre_materia.upper()}")
    print("="*80)

    # 1. Preparar los datos usando la función refactorizada
    df_modelo_final, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)

    print(f"\nSe usarán {len(df_modelo_final)} observaciones completas para el modelo.")

    y = df_modelo_final[variable_y]
    X = df_modelo_final[predictores_finales]

    # 2. División de datos en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Escalar las variables predictoras
    # Es crucial para que la penalización de Lasso funcione correctamente.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Ajustar el modelo LassoCV para encontrar el alpha óptimo
    print("\nBuscando el alpha óptimo con LassoCV (5-fold cross-validation)...")
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    lasso_cv.fit(X_train_scaled, y_train)

    print(f"Alpha óptimo encontrado: {lasso_cv.alpha_:.6f}")

    # 5. Evaluar el modelo final en el conjunto de prueba
    y_pred = lasso_cv.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Evaluación en el Conjunto de Prueba (20%) ---")
    print(f"R-cuadrado (R²): {r2:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")

    # 6. Guardar los coeficientes en Excel
    coefs = pd.Series(lasso_cv.coef_, index=X.columns)
    num_vars_seleccionadas = (coefs != 0).sum()
    print(f"Número de variables seleccionadas por Lasso: {num_vars_seleccionadas} de {len(coefs)}")
    
    coefs_df = coefs.sort_values(ascending=False).to_frame(name='coeficiente_lasso')
    coefs_df.to_excel(writer_lasso, sheet_name=f'Resultados_{nombre_materia}')
    print(f"✅ Coeficientes de Lasso para {nombre_materia} guardados en la hoja '{nombre_materia}' del archivo '{output_excel_path_lasso}'\n")

# Guardar y cerrar el archivo de Excel de Lasso
writer_lasso.close()
print(f"\n🎉 ¡Análisis Lasso completado! Todos los resultados han sido guardados en '{output_excel_path_lasso}'")

#%% Modelo Ridge con Cross-Validation

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Crear un nuevo ExcelWriter para los resultados de Ridge
output_excel_path_ridge = 'resultados_ridge_pisa.xlsx'
writer_ridge = pd.ExcelWriter(output_excel_path_ridge, engine='xlsxwriter')

# Definir un rango de alphas para que RidgeCV pruebe
alphas_ridge = np.logspace(-6, 6, 13)

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f" RIDGE REGRESSION CON CROSS-VALIDATION PARA: {nombre_materia.upper()}")
    print("="*80)

    # 1. Preparar los datos usando la función refactorizada
    df_modelo_final, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)

    print(f"\nSe usarán {len(df_modelo_final)} observaciones completas para el modelo.")

    y = df_modelo_final[variable_y]
    X = df_modelo_final[predictores_finales]

    # 2. División de datos en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Escalar las variables predictoras
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Ajustar el modelo RidgeCV para encontrar el alpha óptimo
    print("\nBuscando el alpha óptimo con RidgeCV (5-fold cross-validation)...")
    ridge_cv = RidgeCV(alphas=alphas_ridge, store_cv_values=True, scoring='neg_root_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train)

    print(f"Alpha óptimo encontrado: {ridge_cv.alpha_:.6f}")

    # 5. Evaluar el modelo final en el conjunto de prueba
    y_pred = ridge_cv.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Evaluación en el Conjunto de Prueba (20%) ---")
    print(f"R-cuadrado (R²): {r2:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")

    # 6. Guardar los coeficientes en Excel
    coefs = pd.Series(ridge_cv.coef_, index=X.columns)
    coefs_df = coefs.sort_values(ascending=False).to_frame(name='coeficiente_ridge')
    coefs_df.to_excel(writer_ridge, sheet_name=f'Resultados_{nombre_materia}')
    print(f"✅ Coeficientes de Ridge para {nombre_materia} guardados en la hoja '{nombre_materia}' del archivo '{output_excel_path_ridge}'\n")

# Guardar y cerrar el archivo de Excel de Ridge
writer_ridge.close()
print(f"\n🎉 ¡Análisis Ridge completado! Todos los resultados han sido guardados en '{output_excel_path_ridge}'")

#%% Heatmap de Correlaciones de Variables Numéricas

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Recrear la lista de predictores numéricos finales para asegurar consistencia
# (Esta lógica es una simplificación de la usada en los modelos, solo para obtener las columnas)
numeric_cols_for_heatmap = []
for col in predictores_filtrados:
    if col not in predictores_dummies:
        numeric_cols_for_heatmap.append(col)

# 2. Seleccionar solo las variables numéricas y los puntajes
puntajes = ['puntaje_matematica', 'puntaje_lengua', 'puntaje_ciencias']
df_heatmap = df[numeric_cols_for_heatmap + puntajes].copy()

# 3. Calcular la matriz de correlación
corr_matrix = df_heatmap.corr()

# 4. Crear una máscara para ocultar la parte superior del heatmap (espejada)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 5. Configurar y generar el gráfico
plt.figure(figsize=(24, 20))

heatmap = sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm', # Usar un colormap divergente
    annot=False, # No mostrar los valores, el gráfico es muy grande
    vmin=-1,
    vmax=1
)

heatmap.set_title('Mapa de Calor de Correlaciones entre Variables Numéricas y Puntajes PISA', 
                  fontdict={'fontsize':18}, 
                  pad=12)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
