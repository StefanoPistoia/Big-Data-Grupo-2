import pandas as pd

# --- 1. Cargar y Preparar los Datos ---
# Cargamos el mismo DataFrame que usaste para los modelos.
df = pd.read_parquet("PISA_LATAM.parquet")
df = df.reset_index()

# --- 2. Replicar la Selección de Predictores ---
# Replicamos la misma lógica de tu script para obtener la lista final de predictores.

# Identificar columnas con más del 50% de valores faltantes
missing_pct = df.isna().sum() / len(df) * 100
columnas_a_eliminar = missing_pct[missing_pct > 50].index.tolist()

# Lista original de predictores
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
    'CREATACT', 'CREATOPN', 'CREATOR'
]

# Listas de exclusión y dummies
afuera = ['COBN_S', 'COBN_M', 'COBN_F', 'LANGN']
predictores_dummies = ['TARDYSD', 'IMMIG', 'ST004D01T']

# Filtrar la lista de predictores
predictores_filtrados = [p for p in predictores if p not in columnas_a_eliminar and p not in afuera]

# Separar en numéricos y categóricos
numeric_predictors = [p for p in predictores_filtrados if p not in predictores_dummies]
categorical_predictors = [p for p in predictores_filtrados if p in predictores_dummies]

# --- 3. Calcular y Mostrar Estadísticas ---

print("\n" + "="*80)
print("📊 ESTADÍSTICAS DESCRIPTIVAS DE PREDICTORES NUMÉRICOS")
print("="*80)
# Usamos .T para transponer la tabla y que sea más fácil de leer
print(df[numeric_predictors].describe().round(2).T.to_string())
print("\n" + "="*80)
print("\n📊 CONTEO DE VALORES PARA PREDICTORES CATEGÓRICOS")
print("="*80)
for col in categorical_predictors:
    print(f"\n--- Variable: {col} ---")
    print(df[col].value_counts(dropna=False))
    print("-"*(len(col) + 16))