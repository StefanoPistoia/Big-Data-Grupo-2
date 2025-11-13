import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# --- 1. PREPARACIÓN DE DATOS Y MODELOS (Lógica adaptada de 'Presentación 2 script.py') ---

# Cargar datos
try:
    df = pd.read_parquet("PISA_LATAM.parquet")
    df = df.reset_index()
except FileNotFoundError:
    print("Error: No se encontró el archivo 'PISA_LATAM.parquet'.")
    print("Asegúrate de ejecutar primero el script 'Sin título0.py' para generarlo.")
    exit()

# Definición de predictores y materias
missing_pct = df.isna().sum() / len(df) * 100
columnas_a_eliminar = missing_pct[missing_pct > 50].index.tolist()

predictores = [
    'ST004D01T','AGE','GRADE','ISCEDP','IMMIG','COBN_S','COBN_M','COBN_F','LANGN','REPEAT','MISSSC','SKIPPING','TARDYSD',
    'EXERPRAC','STUDYHMW','WORKPAY','WORKHOME','EXPECEDU','MATHPREF','MATHEASE','MATHMOT','DURECEC','BSMJ','RELATST',
    'BELONG','BULLIED','FEELSAFE','SCHRISK','PERSEVAGR','CURIOAGR','COOPAGR','EMPATAGR','ASSERAGR','STRESAGR','EMOCOAGR',
    'GROSAGR','INFOSEEK','FAMSUP','DISCLIM','TEACHSUP','COGACRCO','COGACMCO','EXPOFA','EXPO21ST','MATHEFF','MATHEF21','FAMCON',
    'ANXMAT','MATHPERS','CREATEFF','CREATSCH','CREATFAM','CREATAS','CREATOOS','CREATOP','OPENART','IMAGINE','SCHSUST','LEARRES',
    'PROBSELF','FAMSUPSL','FEELLAH','SDLEFF','MISCED','FISCED','HISCED','PAREDINT','BMMJ1','BFMJ2','HISEI','ICTRES','HOMEPOS',
    'ESCS','FCFMLRTY','FLSCHOOL','FLMULTSB','FLFAMILY','ACCESSFP','FLCONFIN','FLCONICT','ACCESSFA','ATTCONFM','FRINFLFM','ICTSCH',
    'ICTAVSCH','ICTHOME','ICTAVHOM','ICTQUAL','ICTSUBJ','ICTENQ','ICTFEED','ICTOUT','ICTWKDY','ICTWKEND','ICTREG','ICTINFO',
    'ICTDISTR','ICTEFFIC','STUBMI','BODYIMA','SOCONPA','LIFESAT','PSYCHSYM','SOCCON','EXPWB','CURSUPP','PQMIMP','PQMCAR','PARINVOL',
    'PQSCHOOL','PASCHPOL','ATTIMMP','PAREXPT','CREATHME','CREATACT','CREATOPN','CREATOR','CNT']

afuera = ['COBN_S', 'COBN_M', 'COBN_F', 'LANGN']
predictores_dummies = ['TARDYSD', 'IMMIG', 'ST004D01T']

predictores_filtrados = [p for p in predictores if p in df.columns and p not in columnas_a_eliminar and p not in afuera and p != 'CNT']

materias = {
    'Matemática': 'puntaje_matematica',
    'Lengua': 'puntaje_lengua',
    'Ciencias': 'puntaje_ciencias'
}

# Calcular puntajes si no existen
for area, prefix in {"matematica": "MATH", "lengua": "READ", "ciencias": "SCIE"}.items():
    if f"puntaje_{area}" not in df.columns:
        pv_cols = [f"PV{i}{prefix}" for i in range(1, 11)]
        df[f"puntaje_{area}"] = df[pv_cols].mean(axis=1)

def prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y):
    columnas_necesarias = predictores_filtrados + [variable_y, 'CNT']
    df_modelo_temp = df[columnas_necesarias].copy()
    numeric_cols_for_model = [p for p in predictores_filtrados if p not in predictores_dummies]
    categorical_cols_to_dummify = [p for p in predictores_filtrados if p in predictores_dummies]
    
    for col in numeric_cols_for_model:
        df_modelo_temp[col] = pd.to_numeric(df_modelo_temp[col], errors='coerce')

    df_dummies_country = pd.get_dummies(df_modelo_temp[['CNT']], columns=['CNT'], drop_first=True, dummy_na=False).astype(int)
    df_dummies_other = pd.get_dummies(df_modelo_temp[categorical_cols_to_dummify].astype('Int64'), columns=categorical_cols_to_dummify, drop_first=True, dummy_na=False).astype(int)

    df_modelo_final = pd.concat([
        df_modelo_temp[numeric_cols_for_model], df_dummies_country, df_dummies_other, df_modelo_temp[[variable_y]]
    ], axis=1)
    df_modelo_final.dropna(inplace=True)
    predictores_finales = numeric_cols_for_model + list(df_dummies_country.columns) + list(df_dummies_other.columns)
    return df_modelo_final, predictores_finales

# Generar las tablas comparativas
tablas_comparativas = {}
for nombre_materia, variable_y in materias.items():
    df_modelo, predictores_finales = prepare_data_for_model(df, predictores_filtrados, predictores_dummies, variable_y)
    y = df_modelo[variable_y]
    X = df_modelo[predictores_finales]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # OLS
    X_train_ols = sm.add_constant(X_train)
    modelo_ols = sm.OLS(y_train, X_train_ols).fit(cov_type='HC1')
    resumen_ols = pd.DataFrame({'coef_ols': modelo_ols.params})

    # Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1).fit(X_train_scaled, y_train)
    coefs_lasso = pd.Series(lasso_cv.coef_, index=X.columns, name='coef_lasso')

    tablas_comparativas[nombre_materia] = resumen_ols.join(coefs_lasso).round(4)

print("✅ Tablas comparativas OLS vs. Lasso generadas en memoria.")

# --- 2. CREACIÓN DEL GRÁFICO ---

variables_interes = ['HOMEPOS', 'PAREDINT']
colores = {'OLS': '#E65747', 'Lasso': '#642C80'}

# --- Consolidar datos en un solo DataFrame ---
datos_consolidados = []
label_map = {
    'HOMEPOS': 'Riqueza del Hogar',
    'PAREDINT': 'Años de educación parentales'
}

for nombre_materia, df_comp in tablas_comparativas.items():
    datos_materia = df_comp.loc[variables_interes].copy()
    datos_materia['materia'] = nombre_materia
    datos_consolidados.append(datos_materia)

df_plot = pd.concat(datos_consolidados).reset_index().rename(columns={'index': 'variable'})
df_plot.rename(columns={'coef_ols': 'OLS', 'coef_lasso': 'Lasso'}, inplace=True)

# Crear etiquetas combinadas para el eje Y
df_plot['label_y'] = df_plot['materia'] + " - " + df_plot['variable'].map(label_map)

# --- Creación del Gráfico Unificado ---
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
fig.suptitle('Comparación de los coeficientes mas importantes OLS vs. Lasso ', fontsize=22, weight='bold')

# Configuración para el gráfico de puntos
y_pos = np.arange(len(df_plot))

# Dibujar las líneas y los puntos para cada variable y materia
for j, row in df_plot.iterrows():
    coef_ols = row['OLS']
    coef_lasso = row['Lasso']
    
    # 1. Dibujar la línea que conecta los puntos
    ax.plot([coef_ols, coef_lasso], [j, j], color='grey', linestyle='-', linewidth=2, zorder=0)
    
    # 2. Dibujar los puntos (scatter plot)
    ax.scatter(coef_ols, j, color=colores['OLS'], s=150, zorder=5, label='OLS' if j == 0 else "")
    ax.scatter(coef_lasso, j, color=colores['Lasso'], s=150, zorder=5, label='Lasso' if j == 0 else "")
    
    # 3. Añadir etiquetas de texto para cada punto
    ax.text(coef_ols, j + 0.15, f'{coef_ols:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')
    ax.text(coef_lasso, j + 0.15, f'{coef_lasso:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')

# Configuración final para toda la figura
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['label_y'], fontsize=14, weight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.axvline(0, color='grey', linestyle='--', linewidth=1)
ax.grid(axis='x', linestyle=':', alpha=0.7)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Valor del Coeficiente', fontsize=16, weight='bold', labelpad=15)
ax.invert_yaxis() # Invertir el eje Y para que Matemática aparezca arriba

# Crear una única leyenda para toda la figura
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

print("\n✅ Gráfico comparativo generado exitosamente.")
