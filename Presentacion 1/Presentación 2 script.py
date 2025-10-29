import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("PISA_LATAM.parquet")
df = df.reset_index()

#modificar ISCEDP
predictores_dummies = ['TARDYSD','IMMIG']
afuera = ['COBN_S','COBN_M','COBN_F','LANGN',]

#Definir la función para calcular los estadísticos por país
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


# --- Filtrado de variables con alto porcentaje de missing ---

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

# 1. Definir las variables predictoras (X) y la variable dependiente (y)

# Lista de variables predictoras que seleccionaste
predictores = [
    'GENDER','AGE', 'GRADE', 'ISCEDP', 'IMMIG', 'COBN_S', 'COBN_M', 'COBN_F', 'LANGN', 
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
    'CREATACT', 'CREATOPN', 'CREATOR'
]
# Lista de variables a excluir explícitamente
afuera = ['COBN_S', 'COBN_M', 'COBN_F', 'LANGN']

# Lista de variables a forzar como dummies
predictores_dummies = ['TARDYSD', 'IMMIG']

# Filtrar la lista de predictores para excluir las columnas con muchos faltantes
predictores_filtrados = [p for p in predictores if p not in columnas_a_eliminar and p not in afuera]


# Variable dependiente
variable_y = 'puntaje_matematica'

# 2. Preparar los datos
# Crear un nuevo DataFrame con solo las columnas necesarias para el modelo
# y la variable de clusterización 'CNT'. Se usa una copia para no modificar el df original.
df_modelo_temp = df[predictores_filtrados + [variable_y, 'CNT']].copy()

# --- Manejo de tipos de datos: Convertir 'object' a numérico o a dummy variables ---

numeric_cols_for_model = []
categorical_cols_to_dummify = []

for col in predictores_filtrados:
    # Intentar convertir a numérico. Si hay errores, se convierten a NaN.
    df_modelo_temp[col] = pd.to_numeric(df_modelo_temp[col], errors='coerce')
    
    # Si es numérica pero tiene pocos valores únicos (sugiere que es categórica codificada)
    # la marcamos para convertir a dummy. Un umbral de 20 valores únicos es una heurística común.
    if col in predictores_dummies:
        categorical_cols_to_dummify.append(col)
    else:
        numeric_cols_for_model.append(col)

# Convertir las columnas categóricas identificadas a variables dummy
if categorical_cols_to_dummify:
    print(f"\nℹ️  Convirtiendo {len(categorical_cols_to_dummify)} variables a formato dummy (one-hot encoding).")
    df_dummies = pd.get_dummies(df_modelo_temp[categorical_cols_to_dummify].astype(str), columns=categorical_cols_to_dummify, drop_first=True)
    df_modelo_temp = pd.concat([df_modelo_temp[numeric_cols_for_model], df_dummies, df_modelo_temp[[variable_y, 'CNT']]], axis=1)
    predictores_finales = numeric_cols_for_model + list(df_dummies.columns)
else:
    predictores_finales = numeric_cols_for_model

# Eliminar filas con datos faltantes en cualquiera de las columnas seleccionadas
df_modelo_temp.dropna(inplace=True)

print(f"\nSe usarán {len(df_modelo_temp)} observaciones completas para el modelo de regresión después de manejar tipos de datos.")

# Definir X e y
y = df_modelo_temp[variable_y]
X = df_modelo_temp[predictores_finales]

# Agregar una constante (intercepto) al modelo
X = sm.add_constant(X)

# 3. Ajustar el modelo OLS con errores estándar clusterizados por país

modelo_ols = sm.OLS(y, X) # Aquí es donde se producía el error
resultados = modelo_ols.fit(cov_type='cluster', cov_kwds={'groups': df_modelo_temp['CNT']})

# 4. Mostrar el resumen de los resultados del modelo
print("\n" + "="*80)
print("📊 RESULTADOS DEL MODELO DE REGRESIÓN OLS (ERRORES CLUSTERIZADOS POR PAÍS)")
print("="*80)
print(resultados.summary())
print("="*80)
