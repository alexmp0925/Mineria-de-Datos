import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
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

df = df.groupby(['fecha', 'linea', 'estacion'], as_index=False)['afluencia'].sum()
df['fecha'] = pd.to_datetime(df['fecha'])
df['fecha_ordinal'] = df['fecha'].map(datetime.toordinal)

catalogo_estaciones = df[['estacion', 'linea']].drop_duplicates()

ultima_fecha = df['fecha'].max()
fechas_futuras = [ultima_fecha + timedelta(days=i) for i in range(1, 181)]

lista_resultados = []

print(f"Iniciando pronóstico individual para {len(catalogo_estaciones)} estaciones...")

for _, fila in catalogo_estaciones.iterrows():
    nom_est = fila['estacion']
    nom_lin = fila['linea']
    
    datos_est = df[(df['estacion'] == nom_est) & (df['linea'] == nom_lin)]
    
    if len(datos_est) < 2:
        continue
        
    modelo_individual = LinearRegression()
    X_train = datos_est[['fecha_ordinal']]
    y_train = datos_est['afluencia']
    modelo_individual.fit(X_train, y_train)
    
    for fecha in fechas_futuras:
        pred = modelo_individual.predict([[fecha.toordinal()]])[0]
        
        lista_resultados.append({
            'linea': nom_lin,
            'estacion': nom_est,
            'fecha': fecha.strftime('%Y-%m-%d'),
            'afluencia_estimada': max(0, int(pred))
        })

pronostico_final = pd.DataFrame(lista_resultados)

pronostico_final = pronostico_final.sort_values(by=['linea', 'estacion', 'fecha'])

pronostico_final.to_csv('Pronostico_6Meses_Completo.csv', index=False, encoding='utf-8-sig')

print(f"\nProceso finalizado.")
print(f"Archivo generado: 'Pronostico_6Meses_Completo.csv'")
print(f"Total de registros: {len(pronostico_final)}")