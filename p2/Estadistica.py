import pandas as pd

#Cargar datos
path = r'C:\Users\conti\Downloads\practica1\Afluencia Diaria del Metro CDMX.csv'
df = pd.read_csv(path)



df['linea'] = df['linea'].astype(str).str.strip()
df['linea'] = df['linea'].str.replace(r'^L.*nea', 'Linea', regex=True, case=False)
df['estacion'] = df['estacion'].astype(str).str.strip()
por_linea = df.groupby('linea')['afluencia'].sum().sort_values(ascending=False)
print(por_linea)


# limpiar datos
df['fecha'] = pd.to_datetime(df['fecha']) 
df = df.dropna()

print("Estadistica descriptiva")
resumen = df['afluencia'].describe()
print(resumen)

print("Estadistica agrupada")

print("Afluencia total por Linea")
por_linea = df.groupby('linea')['afluencia'].sum().sort_values(ascending=False)
print(por_linea)

print("\nLas 5 Estaciones con mayor Afluencia Promedio")
por_estacion = df.groupby('estacion')['afluencia'].mean().sort_values(ascending=False).head(5)
print(por_estacion)

print("\nAfluencia registrada por Año")
por_anio = df.groupby('anio')['afluencia'].max()
print(por_anio)

#guardar 
resumen.to_csv('resumen_estadistico.csv')
print("\nSe guardo 'resumen_estadistico.csv'")