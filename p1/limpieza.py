import pandas as pd

#cargar los datos
df = pd.read_csv(r'C:\Users\conti\Downloads\practica1\Afluencia Diaria del Metro CDMX.csv')
print(df.info())


df['fecha'] = pd.to_datetime(df['fecha'])


print("\nValores nulos encontrados:")
print(df.isnull().sum())


df = df.dropna() 


df['estacion'] = df['estacion'].str.strip()
df['linea'] = df['linea'].str.upper() 

#quitar duplicados
antes = len(df)
df = df.drop_duplicates()
print(f"\nSe eliminaron {antes - len(df)} filas duplicadas")

#guardar
df.to_csv('Afluencia_Metro_Limpio.csv', index=False)
print("\nDataset limpio guardado")