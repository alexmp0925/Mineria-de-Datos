import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ruta_archivo = 'practica5/Afluencia_Metro_Limpio.csv'
carpeta_salida = 'practica5/graficas_lineas'

os.makedirs(carpeta_salida, exist_ok=True)

df = pd.read_csv(ruta_archivo)


lineas = [
    'LINEA 1','LINEA 2','LINEA 3','LINEA 4','LINEA 5','LINEA 6',
    'LINEA 7','LINEA 8','LINEA 9','LINEA A','LINEA B','LINEA 12'
]

for linea_objetivo in lineas:
    df_linea = df[df['linea'] == linea_objetivo]

    if df_linea.empty:
        print(f"No hay datos para {linea_objetivo}")
        continue

    df_anual = df_linea.groupby('anio')['afluencia'].mean().reset_index()

    X = df_anual[['anio']]
    y = df_anual['afluencia']

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='orange', s=100, label=f'Datos reales {linea_objetivo}')
    plt.plot(X, y_pred, color='red', linewidth=3, label='Tendencia Lineal')

    plt.title(f'Analisis de la {linea_objetivo}\n(R² = {r2:.3f})')
    plt.xlabel('Año')
    plt.ylabel('Afluencia Promedio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


    nombre_archivo = f"{linea_objetivo.replace(' ', '_')}.png"
    ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_guardado)

    plt.close()  

    print(f"{linea_objetivo} -> R2: {r2:.4f} | Guardado en: {ruta_guardado}")