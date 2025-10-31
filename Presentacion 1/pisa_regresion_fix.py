import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Cargar datos
df = pd.read_parquet("PISA_LATAM.parquet")
df = df.reset_index()

# --- Calcular puntajes promedio ---
for area, prefix in {"matematica": "MATH", "lengua": "READ", "ciencias": "SCIE"}.items():
    pv_cols = [f"PV{i}{prefix}" for i in range(1, 11)]
    df[f"puntaje_{area}"] = df[pv_cols].mean(axis=1)

print(df[["CNT", "puntaje_matematica", "puntaje_lengua", "puntaje_ciencias"].head()])

# --- Preparar modelo OLS ---
missing_pct = df.isna().sum() / len(df) * 100
columnas_a_eliminar = missing_pct[missing_pct > 50].index.tolist()

predictores = [
    'ST004D01T','AGE','GRADE','ISCEDP','IMMIG','COBN_S','COBN_M','COBN_F','LANGN','REPEAT','MISSSC','SKIPPING','TARDYSD',
    'EXERPRAC','STUDYHMW','WORKPAY','WORKHOME','EXPECEDU','MATHPREF','MATHEASE','MATHMOT','DURECEC','BSMJ','SISCO','RELATST',
    'BELONG','BULLIED','FEELSAFE','SCHRISK','PERSEVAGR','CURIOAGR','COOPAGR','EMPATAGR','ASSERAGR','STRESAGR','EMOCOAGR',
    'GROSAGR','INFOSEEK','FAMSUP','DISCLIM','TEACHSUP','COGACRCO','COGACMCO','EXPOFA','EXPO21ST','MATHEFF','MATHEF21','FAMCON',
    'ANXMAT','MATHPERS','CREATEFF','CREATSCH','CREATFAM','CREATAS','CREATOOS','CREATOP','OPENART','IMAGINE','SCHSUST','LEARRES',
    'PROBSELF','FAMSUPSL','FEELLAH','SDLEFF','MISCED','FISCED','HISCED','PAREDINT','BMMJ1','BFMJ2','HISEI','ICTRES','HOMEPOS',
    'ESCS','FCFMLRTY','FLSCHOOL','FLMULTSB','FLFAMILY','ACCESSFP','FLCONFIN','FLCONICT','ACCESSFA','ATTCONFM','FRINFLFM','ICTSCH',
    'ICTAVSCH','ICTHOME','ICTAVHOM','ICTQUAL','ICTSUBJ','ICTENQ','ICTFEED','ICTOUT','ICTWKDY','ICTWKEND','ICTREG','ICTINFO',
    'ICTDISTR','ICTEFFIC','STUBMI','BODYIMA','SOCONPA','LIFESAT','PSYCHSYM','SOCCON','EXPWB','CURSUPP','PQMIMP','PQMCAR','PARINVOL',
    'PQSCHOOL','PASCHPOL','ATTIMMP','PAREXPT','CREATHME','CREATACT','CREATOPN','CREATOR','CNT']

afuera = ['COBN_S', 'COBN_M', 'COBN_F', 'LANGN']
predictores_dummies = ['TARDYSD', 'IMMIG', 'ST004D01T', 'CNT']

# --- Verificar columnas existentes ---
predictores_presentes = [p for p in predictores if p in df.columns]
predictores_faltantes = [p for p in predictores if p not in df.columns]

print("\n" + "="*80)
print("🔍 Verificación de variables en el DataFrame:")
print(f"✅ Variables encontradas: {len(predictores_presentes)}")
print(f"❌ Variables faltantes: {len(predictores_faltantes)}")
if predictores_faltantes:
    print("\nListado de variables faltantes:")
    for col in predictores_faltantes:
        print(f" - {col}")
print("="*80)

predictores_filtrados = [p for p in predictores_presentes if p not in columnas_a_eliminar and p not in afuera]

# --- Materias ---
materias = {
    'Matemática': 'puntaje_matematica',
    'Lengua': 'puntaje_lengua',
    'Ciencias': 'puntaje_ciencias'
}

output_excel_path = 'resultados_regresion_pisa.xlsx'
writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')

for nombre_materia, variable_y in materias.items():
    print("\n" + "="*80)
    print(f"📊 EJECUTANDO MODELO DE REGRESIÓN OLS PARA: {nombre_materia.upper()}")
    print("="*80)

    df_modelo_temp = df[predictores_filtrados + [variable_y, 'CNT']].copy()

    numeric_cols_for_model = []
    categorical_cols_to_dummify = []

    for col in predictores_filtrados:
        if col not in df_modelo_temp.columns:
            print(f"⚠️ Columna {col} no está en df_modelo_temp")
            continue
        try:
            df_modelo_temp[col] = pd.to_numeric(df_modelo_temp[col], errors='coerce')
        except Exception as e:
            print(f"❌ Error al convertir {col}: {type(e).__name__} - {e}")
        if col in predictores_dummies:
            categorical_cols_to_dummify.append(col)
        else:
            numeric_cols_for_model.append(col)

    if categorical_cols_to_dummify:
        print(f"\nℹ️  Convirtiendo {len(categorical_cols_to_dummify)} variables a formato dummy.")
        country_col = ['CNT']
        other_dummy_cols = [col for col in categorical_cols_to_dummify if col != 'CNT']

        df_dummies_country = pd.get_dummies(df_modelo_temp[country_col], columns=country_col, drop_first=True).astype(int)
        df_dummies_temp_other = df_modelo_temp[other_dummy_cols].astype('Int64')
        df_dummies_other = pd.get_dummies(df_dummies_temp_other.astype(str), columns=other_dummy_cols, drop_first=True).astype(int)

        df_modelo_temp = pd.concat([
            df_modelo_temp[numeric_cols_for_model],
            df_dummies_country,
            df_dummies_other,
            df_modelo_temp[[variable_y]]
        ], axis=1)

        predictores_finales = numeric_cols_for_model + list(df_dummies_country.columns) + list(df_dummies_other.columns)
    else:
        predictores_finales = numeric_cols_for_model

    df_modelo_temp.dropna(inplace=True)
    print(f"\nSe usarán {len(df_modelo_temp)} observaciones completas para el modelo.")

    y = df_modelo_temp[variable_y]
    X = df_modelo_temp[predictores_finales]
    X = sm.add_constant(X)

    modelo_ols = sm.OLS(y, X)
    resultados = modelo_ols.fit(cov_type='HC1')

    print(resultados.summary())

    resumen_df = pd.read_html(resultados.summary().tables[1].as_html(), header=0, index_col=0)[0]
    resumen_df.to_excel(writer, sheet_name=f'Resultados_{nombre_materia}')
    print(f"✅ Resultados para {nombre_materia} guardados.")

writer.close()
print(f"\n🎉 ¡Análisis completado! Resultados guardados en '{output_excel_path}'")
