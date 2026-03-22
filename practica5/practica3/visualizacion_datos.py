import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# rutas
dir_actual = os.path.dirname(__file__)
path_data = os.path.join(dir_actual, 'Afluencia_Metro_Limpio.csv')
output_folder = os.path.join(dir_actual, 'graficas')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_csv(path_data, encoding='latin-1')

df['linea'] = df['linea'].astype(str).str.strip().str.upper()
df['linea'] = df['linea'].str.replace(r'[^\w\s]', '', regex=True)

df['linea'] = df['linea'].str.extract(r'(\d+|A|B)$', expand=False)
df['linea'] = 'LINEA ' + df['linea']

sns.set_theme(style="whitegrid")
print("Generando visualizaciones...")

#Diagrama de barras
plt.figure(figsize=(12, 6))
top_estaciones = df.groupby('estacion')['afluencia'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_estaciones.values, y=top_estaciones.index, palette='viridis')
plt.title('Las 10 estaciones con mayor afluencia promedio')
plt.savefig(os.path.join(output_folder, '01_barras_top_estaciones.png'))
plt.close()

# Diagrama circular
plt.figure(figsize=(10, 10))
por_linea = df.groupby('linea')['afluencia'].sum()
plt.pie(por_linea, labels=por_linea.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
plt.title('Distribucion de afluencia por linea')
plt.savefig(os.path.join(output_folder, '02_circular_proporcion_lineas.png'))
plt.close()

#histogramas
lineas = df['linea'].unique()
for linea in lineas[:12]:
    plt.figure(figsize=(8, 5))
    data_linea = df[df['linea'] == linea]
    sns.histplot(data_linea['afluencia'], kde=True, color='skyblue')
    plt.title(f'Histograma de frecuencias: {linea}')
    plt.xlabel('Cantidad de Pasajeros')
    plt.savefig(os.path.join(output_folder, f'03_histograma_{linea}.png'))
    plt.close()

# Boxplot
plt.figure(figsize=(14, 7))
sns.boxplot(x='linea', y='afluencia', data=df)
plt.xticks(rotation=45)
plt.title('Comparacion de dispersion y Outliers por Linea')
plt.savefig(os.path.join(output_folder, '04_boxplot_comparativo.png'))
plt.close()

# Dispersion
df_sample = df.sample(n=1000) 
plt.figure(figsize=(12, 6))
df_sample['fecha'] = pd.to_datetime(df_sample['fecha'])
sns.scatterplot(data=df_sample, x='fecha', y='afluencia', hue='linea', alpha=0.7, s=40)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
plt.xticks(rotation=45) 
plt.title('Dispersion de afluencia a travrs del Tiempo')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Líneas")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_dispersion_tiempo.png'))
plt.close()




