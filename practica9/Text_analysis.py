import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import unicodedata
import re
import os  

def limpiar_texto_mejorado(texto):
    if not isinstance(texto, str): return ""
    
    try:
        texto = texto.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass 

    texto = unicodedata.normalize('NFKD', texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.upper().strip()
    texto = re.sub(r'[^A-Z\s]', '', texto)
    return " ".join(texto.split())

df = pd.read_csv('C:\\Users\\conti\\OneDrive\\Desktop\\220326\\MD\\practica9\\Afluencia_Metro_Limpio.csv', encoding='latin-1')

df['estacion_limpia'] = df['estacion'].apply(limpiar_texto_mejorado)


carpeta_destino = "practica9"
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)


resumen_afluencia = df.groupby('estacion_limpia')['afluencia'].sum().to_dict()


wordcloud = WordCloud(
    width=1200, 
    height=600,
    background_color='white',
    colormap='inferno',
    stopwords={'DE', 'LA', 'EL', 'LOS', 'LAS', 'DEL', 'LINEA', 'ESTACION'}
).generate_from_frequencies(resumen_afluencia)


plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title(f'Nube de Palabras Ponderada por Afluencia')


ruta_final = os.path.join(carpeta_destino, "nube_ponderada_limpia.png")
wordcloud.to_file(ruta_final)

ranking = sorted(resumen_afluencia.items(), key=lambda x: x[1], reverse=True)

print("Top 5 estaciones con mas peso en la nube:")
for estacion, total in ranking[:5]:
    print(f"{estacion}: {total:,} pasajeros acumulados")
print(f"Proceso completado. Imagen guardada en: {ruta_final}")
plt.show()