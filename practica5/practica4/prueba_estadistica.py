import pandas as pd
from scipy import stats
import os
import scikit_posthocs as sp

dir_actual = os.path.dirname(__file__)
path_data = os.path.join(dir_actual, 'Afluencia_Metro_Limpio.csv')
df = pd.read_csv(path_data, encoding='latin-1')
df['linea'] = df['linea'].astype(str).str.strip().str.upper()
df['linea'] = df['linea'].str.extract(r'(\d+|A|B)$', expand=False)
df['linea'] = 'LINEA ' + df['linea']
df = df.dropna(subset=['linea'])

print("Demostracion estadistica")

lineas = sorted(df['linea'].unique())
grupos = [df[df['linea'] == l]['afluencia'] for l in lineas]

h_stat, p_kruskal = stats.kruskal(*grupos)

if p_kruskal < 0.05:
    print(f"Kruskal-Wallis p-value: {p_kruskal:.4e}")

print("\nResumen de Afluencia Promedio por Linea:")
resumen = df.groupby('linea')['afluencia'].mean().sort_values(ascending=False)
print(resumen)

print("\nComparartiva de todasd las estaciones")
p_values_matrix = sp.posthoc_dunn(df, val_col='afluencia', group_col='linea', p_adjust='holm')
print(p_values_matrix)
print("NOTA: '0.000000e+00' quiere decir que el valor es extremadamente bajo")


print("\nInterpretacion de la comparativa:")


comparaciones = p_values_matrix.stack().reset_index()
comparaciones.columns = ['Linea x', 'Linea y', 'p_value']

similares = comparaciones[(comparaciones['p_value'] > 0.05) & (comparaciones['Linea x'] != comparaciones['Linea y'])]

if similares.empty:
    print("No hay lineas similares entre si. Cada linea tiene un comportamiento unico y diferente")


