import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("PISA_LATAM.parquet")
df = df.reset_index()



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
columnas_a_eliminar = missing_pct[missing_pct > 50].index.tolist()

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
    'DURECEC', 'BSMJ', 'RELATST', 'BELONG', 'BULLIED', 'FEELSAFE', 
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

#%% Modelo Random Forest con evaluación OOB

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f"🌳 EJECUTANDO RANDOM FOREST PARA: {nombre_materia.upper()}")
    print("="*80)

    # 1. Preparar los datos usando la función refactorizada
    df_modelo_final, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)

    print(f"\nSe usarán {len(df_modelo_final)} observaciones completas para el modelo de Random Forest.")

    y = df_modelo_final[variable_y]
    X = df_modelo_final[predictores_finales]

    # 2. División de datos en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Tamaño del conjunto de Entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Tamaño del conjunto de Prueba: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # 3. Configurar y entrenar el modelo Random Forest
    # oob_score=True calcula el score en las muestras "Out-of-Bag", una buena estimación del rendimiento.
    # n_jobs=-1 usa todos los procesadores para acelerar el entrenamiento.
    print("\nEntrenando el modelo Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, oob_score=True, n_jobs=-1, max_features='sqrt', min_samples_leaf=4)
    rf_model.fit(X_train, y_train)

    # 4. Evaluar el modelo en el conjunto de prueba
    y_pred_final = rf_model.predict(X_test)

    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
    r2_final = r2_score(y_test, y_pred_final)

    print("\n--- Resultados de la Evaluación Final en el Conjunto de Prueba (20%) ---")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_final:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2_final:.4f}")
    print(f"Out-of-Bag (OOB) Score (R² estimado sobre datos no vistos durante el entrenamiento): {rf_model.oob_score_:.4f}")
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

#%% Modelo Elastic Net con Cross-Validation

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Crear un nuevo ExcelWriter para los resultados de Elastic Net
output_excel_path_elasticnet = 'resultados_elasticnet_pisa.xlsx'
writer_elasticnet = pd.ExcelWriter(output_excel_path_elasticnet, engine='xlsxwriter')

# Definir un rango de l1_ratios para que ElasticNetCV pruebe
# l1_ratio = 1 es Lasso, l1_ratio = 0 es Ridge (casi, alpha=0 no es exactamente lo mismo)
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f" ELASTIC NET REGRESSION CON CROSS-VALIDATION PARA: {nombre_materia.upper()}")
    print("="*80)

    # 1. Preparar los datos usando la función refactorizada
    df_modelo_final, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)

    print(f"\nSe usarán {len(df_modelo_final)} observaciones completas para el modelo.")

    y = df_modelo_final[variable_y]
    X = df_modelo_final[predictores_finales]

    # 2. División de datos en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Escalar las variables predictoras
    # Es crucial para que la penalización funcione correctamente.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Ajustar el modelo ElasticNetCV para encontrar el alpha y l1_ratio óptimos
    print("\nBuscando los hiperparámetros óptimos con ElasticNetCV (5-fold cross-validation)...")
    elasticnet_cv = ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    elasticnet_cv.fit(X_train_scaled, y_train)

    print(f"Alpha óptimo encontrado: {elasticnet_cv.alpha_:.6f}")
    print(f"L1 Ratio óptimo encontrado: {elasticnet_cv.l1_ratio_:.2f}")

    # 5. Evaluar el modelo final en el conjunto de prueba
    y_pred = elasticnet_cv.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Evaluación en el Conjunto de Prueba (20%) ---")
    print(f"R-cuadrado (R²): {r2:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")

    # 6. Guardar los coeficientes en Excel
    coefs = pd.Series(elasticnet_cv.coef_, index=X.columns)
    num_vars_seleccionadas = (coefs != 0).sum()
    print(f"Número de variables seleccionadas por Elastic Net: {num_vars_seleccionadas} de {len(coefs)}")
    
    coefs_df = coefs.sort_values(ascending=False).to_frame(name='coeficiente_elasticnet')
    coefs_df.to_excel(writer_elasticnet, sheet_name=f'Resultados_{nombre_materia}')
    print(f"✅ Coeficientes de Elastic Net para {nombre_materia} guardados en la hoja '{nombre_materia}' del archivo '{output_excel_path_elasticnet}'\n")

# Guardar y cerrar el archivo de Excel de Elastic Net
writer_elasticnet.close()
print(f"\n🎉 ¡Análisis Elastic Net completado! Todos los resultados han sido guardados en '{output_excel_path_elasticnet}'")

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
   
 # --- Comprobación de Multicolinealidad (VIF) ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\nCalculando el Factor de Inflación de la Varianza (VIF) para las variables predictoras...")
print("Un VIF alto (generalmente > 10) sugiere multicolinealidad.")

    # El cálculo de VIF puede ser computacionalmente intensivo con muchas variables.
    # Lo calculamos sobre el DataFrame final antes de la división train/test.
X_vif = df_modelo_final[predictores_finales]
    
    # Añadir una constante para el cálculo de VIF, como en un modelo de regresión
X_vif_const = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
    
    # Mostrar las 10 variables con el VIF más alto, excluyendo la constante
print("\n--- Top 10 Variables con Mayor VIF ---")
print(vif_data.sort_values('VIF', ascending=False).drop(vif_data[vif_data['feature'] == 'const'].index).head(10))
print("="*80)

#%% Análisis de Linealidad de Predictores vs. Puntaje de Matemática

print("\n" + "="*80)
print("🔎 ANALIZANDO LA LINEALIDAD DE LOS PREDICTORES CON EL PUNTAJE DE MATEMÁTICA")
print("="*80)

# 1. Identificar predictores numéricos (excluyendo los que se convirtieron en dummies)
numeric_predictors = [p for p in predictores_filtrados if p not in predictores_dummies and p != 'CNT']

# 2. Análisis Estadístico: Calcular Correlación de Pearson
print("\n--- Coeficientes de Correlación de Pearson con 'puntaje_matematica' ---")
print("Mide la fuerza de la relación LINEAL. Valores cercanos a 0 indican una relación lineal débil.")

# Crear un DataFrame temporal con las variables de interés y eliminar NaNs para el cálculo
df_corr = df[numeric_predictors + ['puntaje_matematica']].dropna()

correlations = df_corr[numeric_predictors].corrwith(df_corr['puntaje_matematica'])

# Mostrar las 15 correlaciones más fuertes (positivas y negativas)
correlations_abs_sorted = correlations.abs().sort_values(ascending=False)
print("\nTop 15 correlaciones más fuertes (en valor absoluto):")
print(correlations.loc[correlations_abs_sorted.head(15).index].to_string())

# 3. Análisis Visual: Generar Grids de Gráficos de Dispersión
print("\nGenerando gráficos de dispersión para visualizar la linealidad...")

n_predictors = len(numeric_predictors)
plots_per_grid = 16 # 4x4 grid

for i in range(0, n_predictors, plots_per_grid):
    chunk_predictors = numeric_predictors[i:i + plots_per_grid]
    
    # Determinar el tamaño de la grilla
    n_cols = 4
    n_rows = (len(chunk_predictors) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() # Aplanar el array de ejes para iterar fácilmente

    for j, predictor in enumerate(chunk_predictors):
        sns.regplot(data=df, x=predictor, y='puntaje_matematica', ax=axes[j], 
                    scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
        axes[j].set_title(f'{predictor} vs. Matemática')
        axes[j].set_xlabel(predictor)
        axes[j].set_ylabel('Puntaje Matemática')

    # Ocultar ejes no utilizados
    for k in range(j + 1, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.suptitle(f'Análisis de Linealidad (Parte {i//plots_per_grid + 1})', fontsize=22, y=1.02)
    plt.show()

print("\n✅ Análisis de linealidad completado.")
   
 # --- Comprobación de Multicolinealidad (VIF) ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\nCalculando el Factor de Inflación de la Varianza (VIF) para las variables predictoras...")
print("Un VIF alto (generalmente > 10) sugiere multicolinealidad.")

    # El cálculo de VIF puede ser computacionalmente intensivo con muchas variables.
    # Lo calculamos sobre el DataFrame final antes de la división train/test.
X_vif = df_modelo_final[predictores_finales]
    
    # Añadir una constante para el cálculo de VIF, como en un modelo de regresión
X_vif_const = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
    
    # Mostrar las 10 variables con el VIF más alto, excluyendo la constante
print("\n--- Top 10 Variables con Mayor VIF ---")
print(vif_data.sort_values('VIF', ascending=False).drop(vif_data[vif_data['feature'] == 'const'].index).head(10))
print("="*80)

#%%
import pandas as pd

sheet1 = pd.read_excel('resultados_regresion_pisa.xlsx', sheet_name='Resultados_Matemática', index_col=0)
sheet2 = pd.read_excel('resultados_lasso_pisa.xlsx', sheet_name='Resultados_Matemática', index_col=0)

merged = sheet1.merge(sheet2, left_index=True, right_index=True, how='left')
merged.to_excel('merged_pre.xlsx')


#%% Tablas Comparativas OLS vs. Lasso en el Environment

print("\n" + "="*80)
print("📊 GENERANDO TABLAS COMPARATIVAS OLS vs. LASSO")
print("="*80)

# Diccionario para almacenar las tablas finales en el environment
tablas_comparativas = {}

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f"🔄  Generando tabla comparativa para: {nombre_materia.upper()}")
    print("="*80)

    # 1. Preparar datos
    df_modelo, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)
    y = df_modelo[variable_y]
    X = df_modelo[predictores_finales]

    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Modelo OLS
    X_train_ols = sm.add_constant(X_train)
    modelo_ols = sm.OLS(y_train, X_train_ols).fit(cov_type='HC1')
    resumen_ols = pd.DataFrame({
        'coef_ols': modelo_ols.params,
        'std_err_ols': modelo_ols.bse,
        'p_value_ols': modelo_ols.pvalues
    })

    # 4. Modelo Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    lasso_cv.fit(X_train_scaled, y_train)
    coefs_lasso = pd.Series(lasso_cv.coef_, index=X.columns, name='coef_lasso')

    # 5. Fusionar y mostrar resultados
    df_comparativo = resumen_ols.join(coefs_lasso).round(4)
    tablas_comparativas[nombre_materia] = df_comparativo

    print(f"\n--- Tabla Comparativa de Coeficientes: {nombre_materia} ---")
    print(df_comparativo.to_string())
    print("="*80 + "\n")

print("\n🎉 ¡Análisis comparativo completado!")
print("Las tablas están disponibles en el diccionario 'tablas_comparativas'.")


#%% Gráficos comparativos de coeficientes OLS vs. Lasso

print("\n" + "="*80)
print("📈 GENERANDO GRÁFICOS COMPARATIVOS DE COEFICIENTES")
print("="*80)

# Corregí un pequeño error en la lista (faltaba una coma entre CREATAS y GROSAGR)
top_coefs = ['MATHEFF', 'ST004D01T_2', 'HISCED', 'EXERPRAC', 'WORKPAY', 
             'FAMCON', 'BMMJ1', 'REPEAT', 'CREATAS', 'GROSAGR']

# Crear un diccionario para mapear los nombres de las variables a etiquetas más claras
label_map = {
    'MATHEFF': 'Autoeficacia Matemática',
    'ST004D01T_2': 'Varón respecto a mujer',
    'HISCED': 'Educación de padres',
    'EXERPRAC': 'Practica deporte',
    'WORKPAY': 'Trabajo pago por semana',
    'FAMCON': 'Familiaridad conceptos matemáticos',
    'BMMJ1': 'Nivel ocupacional madre',
    'REPEAT': 'Repitió',
    'CREATAS': 'Actividades creativas en la escuela',
    'GROSAGR': 'Mentalidad de crecimiento'
}
# Iterar sobre cada materia para crear un gráfico distinto
for nombre_materia, variable_y in materias.items():
    
    # 1. Obtener la tabla comparativa correspondiente
    df_comp = tablas_comparativas[nombre_materia]
    
    # 2. Filtrar solo los coeficientes de interés y preparar para graficar
    df_plot = df_comp.loc[df_comp.index.isin(top_coefs)].copy()
    df_plot = df_plot[['coef_ols', 'coef_lasso']]
    
    # 3. Transformar de formato ancho a largo para seaborn
    df_plot_long = df_plot.reset_index().melt(
        id_vars='index', 
        value_vars=['coef_ols', 'coef_lasso'],
        var_name='modelo',
        value_name='coeficiente'
    )
    df_plot_long.rename(columns={'index': 'variable'}, inplace=True)
    
    # Aplicar el mapeo para usar las nuevas etiquetas
    df_plot_long['variable'] = df_plot_long['variable'].map(label_map)

    # 4. Crear el gráfico
    plt.figure(figsize=(14, 8))
    # Ajuste: Usamos dodge=True para separar por modelo, pero quitamos el jitter
    # para que los puntos queden perfectamente alineados en dos columnas.
    ax = sns.stripplot(data=df_plot_long, x='variable', y='coeficiente', hue='modelo',
                       palette={'coef_ols': 'blue', 'coef_lasso': 'red'},
                       dodge=True, jitter=False, size=12, alpha=0.8)
    
    # 5. Añadir el valor de los coeficientes sobre cada punto
    # 5. Añadir el valor de los coeficientes sobre cada punto con desplazamiento dinámico
    # Calcular un desplazamiento dinámico basado en la escala del eje Y para evitar solapamientos
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    dynamic_offset = y_range * 0.02  # Usar un 2% del rango del eje Y como desplazamiento

    for p in ax.collections:
        for offset in p.get_offsets():
            x, y = offset
            # Aplicar el desplazamiento dinámico
            ax.text(x, y + dynamic_offset if y >= 0 else y - dynamic_offset, f'{y:.2f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)

    plt.axhline(0, color='grey', linestyle='--', linewidth=1) # Línea en y=0
    plt.title(f'Comparación de Coeficientes OLS vs. Lasso para {nombre_materia}', fontsize=16)
    plt.ylabel('Valor del Coeficiente', fontsize=12)
    plt.xlabel('Variable Predictora', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.legend(title='Modelo')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()



# --- Comprobación de Multicolinealidad (VIF) ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\nCalculando el Factor de Inflación de la Varianza (VIF) para las variables predictoras...")
print("Un VIF alto (generalmente > 10) sugiere multicolinealidad.")

    # El cálculo de VIF puede ser computacionalmente intensivo con muchas variables.
    # Lo calculamos sobre el DataFrame final antes de la división train/test.
X_vif = df_modelo_final[predictores_finales]
    
    # Añadir una constante para el cálculo de VIF, como en un modelo de regresión
X_vif_const = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
    
    # Mostrar las 10 variables con el VIF más alto, excluyendo la constante
print("\n--- Top 10 Variables con Mayor VIF ---")
print(vif_data.sort_values('VIF', ascending=False).drop(vif_data[vif_data['feature'] == 'const'].index).head(10))
print("="*80)

#%% Gráficos de Densidad de Puntajes por Materia
print("📈 GENERANDO GRÁFICOS DE DENSIDAD DE PUNTAJES")
print("="*80)

# Diccionario para mapear códigos de país a nombres completos
nombres_paises = {
    'ARG': 'Argentina', 'BRA': 'Brasil', 'CHL': 'Chile', 'COL': 'Colombia', 
    'CRI': 'Costa Rica', 'DOM': 'Rep. Dominicana', 'GTM': 'Guatemala', 
    'MEX': 'México', 'PAN': 'Panamá', 'PER': 'Perú', 'PRY': 'Paraguay', 
    'SLV': 'El Salvador', 'URY': 'Uruguay'
}
    # 1. Calcular el puntaje promedio por país para la materia actual
    
# --- Crear una figura con 3 subplots (uno para cada materia) ---
# 1 fila, 3 columnas. `sharey=True` hace que todos los gráficos compartan el mismo eje Y para una comparación más fácil.
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
fig.suptitle('Distribución de Puntajes por Materia: Mejor vs. Peor País en LATAM', fontsize=20, weight='bold')

# Iterar sobre cada materia y su eje correspondiente en la figura
for i, (nombre_materia, variable_y) in enumerate(materias.items()):

    puntajes_por_pais = df.groupby('CNT')[variable_y].mean()
    pais_mejor = puntajes_por_pais.idxmax()
    puntaje_mejor = puntajes_por_pais.max()
    pais_peor = puntajes_por_pais.idxmin()
    puntaje_peor = puntajes_por_pais.min()

    # Obtener los nombres completos para la leyenda
    nombre_mejor = nombres_paises.get(pais_mejor, pais_mejor)
    nombre_peor = nombres_paises.get(pais_peor, pais_peor)

    print(f"\nAnálisis para: {nombre_materia}")
    print(f"  - Mejor país: {nombre_mejor} ({pais_mejor}) - Promedio: {puntaje_mejor:.2f}")
    print(f"  - Peor país:  {nombre_peor} ({pais_peor}) - Promedio: {puntaje_peor:.2f}")

    # 3. Crear el gráfico de densidad
    # La clave es pasar el eje `ax=ax` a TODAS las llamadas de sns.kdeplot
    ax = axes[i]
    sns.kdeplot(data=df, x=variable_y, label='LATAM (General)', color='gray', linewidth=2, fill=True, alpha=0.1, ax=ax)
    
    # Curva de densidad para el país con mejor desempeño
    sns.kdeplot(data=df[df['CNT'] == pais_mejor], x=variable_y, label=f'Mejor: {nombre_mejor}', color='#C2297A', linewidth=2.5, linestyle='--', ax=ax)
    
    # Curva de densidad para el país con peor desempeño
    # CORRECCIÓN: Se añade `ax=ax` para que se dibuje en el subplot correcto.
    sns.kdeplot(data=df[df['CNT'] == pais_peor], x=variable_y, label=f'Peor: {nombre_peor}', color='#FAD958', linewidth=2.5, linestyle='--', ax=ax)

    # 4. Añadir detalles y mejorar la estética del gráfico
    ax.set_title(f'{nombre_materia}', fontsize=16, weight='bold')
    ax.set_xlabel('Puntaje', fontsize=12)
    ax.legend().set_visible(False) # Ocultar las leyendas individuales
    ax.grid(axis='y', linestyle=':', alpha=0.6)

axes[0].set_ylabel('Densidad', fontsize=12) # Poner la etiqueta del eje Y solo en el primer gráfico

# Crear una única leyenda para toda la figura en la parte superior
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.05), ncol=3, fontsize=13)

plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Ajustar el layout para que el supertítulo y la leyenda no se solapen
plt.show()

print("\n" + "="*80)
print("📈 GENERANDO GRÁFICOS DE DENSIDAD DE PUNTAJES")
print("="*80)
