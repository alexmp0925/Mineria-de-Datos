import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import unicodedata
import re

def super_limpieza(texto, es_linea=False):
    if not isinstance(texto, str): return texto
    

    try:
        texto = texto.encode('latin-1').decode('utf-8')
    except:
        pass
    

    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.upper().strip()

    if es_linea:
        match = re.search(r'(\d+|[AB])$', texto)
        if match:
            return f"LINEA {match.group(1)}"
    
    texto = re.sub(r'[^A-Z0-9\s/]', '', texto)
    return texto.strip()

df = pd.read_csv('Afluencia_Metro_Limpio.csv', encoding='latin-1')

df['linea'] = df['linea'].apply(lambda x: super_limpieza(x, es_linea=True))
df['estacion'] = df['estacion'].apply(lambda x: super_limpieza(x, es_linea=False))


df = df.groupby(['fecha', 'anio', 'linea', 'estacion'], as_index=False)['afluencia'].sum()

limite_bajo = df['afluencia'].quantile(0.33)
limite_medio = df['afluencia'].quantile(0.66)

def categorizar(valor):
    if valor <= limite_bajo: return 'BAJA'
    elif valor <= limite_medio: return 'MEDIA'
    else: return 'ALTA'

df['nivel_real'] = df['afluencia'].apply(categorizar)

le_linea = LabelEncoder()
le_estacion = LabelEncoder()

df['linea_id'] = le_linea.fit_transform(df['linea'])
df['estacion_id'] = le_estacion.fit_transform(df['estacion'])

X = df[['linea_id', 'estacion_id', 'anio']]
y = df['nivel_real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train_scaled, y_train)

X_all_scaled = scaler.transform(X)
df['prediccion_afluencia'] = knn.predict(X_all_scaled)

reporte = df[['linea', 'estacion', 'prediccion_afluencia']].drop_duplicates(subset=['linea', 'estacion'])
reporte = reporte.sort_values(by=['linea', 'estacion'])

print("\n" + "="*40)
print("--- Informe de Clasificación (DATOS UNIFICADOS) ---")
print(classification_report(y_test, knn.predict(X_test_scaled)))
print("="*40)

pd.set_option('display.max_rows', None)
print("\n--- Clasificación Final del Sistema Metro (Limpio) ---")
print(reporte.to_string(index=False))

reporte.to_csv('Reporte_Final_Metro_CDMX.csv', index=False, encoding='utf-8-sig')