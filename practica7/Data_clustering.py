import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import unicodedata
import re

def limpiar_texto(texto, es_linea=False):
    if not isinstance(texto, str): return texto
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    texto = texto.upper().strip()
    if es_linea:
        match = re.search(r'(\d+|[AB])$', texto)
        if match: return f"LINEA {match.group(1)}"
    return re.sub(r'[^A-Z0-9\s/]', '', texto).strip()

df = pd.read_csv('Afluencia_Metro_Limpio.csv', encoding='latin-1')
df['linea'] = df['linea'].apply(lambda x: limpiar_texto(x, es_linea=True))
df['estacion'] = df['estacion'].apply(limpiar_texto)

df_grouped = df.groupby(['linea', 'estacion'], as_index=False)['afluencia'].mean()


X = df_grouped[['afluencia']] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inercia = []
K_range = range(1, 10)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inercia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inercia, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia (Suma de distancias al cuadrado)')
plt.title('Método del Codo para la Afluencia del Metro')
plt.show()

k_optimo = 3
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_grouped['cluster'] = kmeans.fit_predict(X_scaled)

centroides = df_grouped.groupby('cluster')['afluencia'].agg(['mean', 'min', 'max', 'count']).sort_values(by='mean')
print("\n--- Análisis de los Clusters Encontrados ---")
print(centroides)

df_grouped.to_csv('Agrupamiento_Metro_Resultados.csv', index=False)
print("\nArchivo 'Agrupamiento_Metro_Resultados.csv' generado.")