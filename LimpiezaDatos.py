import pandas as pd

# Leer el archivo CSV
data = pd.read_csv("dataset2.csv",sep=";" , encoding="latin-1")
["SERENAZGO","LIMPIEZA PUBLICA","PARQUES Y JARDINES","TRANSITO","MANTENIMIENTO","GERENCIA DE SALUD","DEFENSA CIVIL","PARTICIPACIÓN VECINAL"]
# Diccionario de correspondencia entre categorías y valores numéricos
categorias = {
    "SERENAZGO": 1,
    "LIMPIEZA PUBLICA": 2,
    "PARQUES Y JARDINES": 3,
    "TRANSITO":4,
    "MANTENIMIENTO":5,
    "GERENCIA DE SALUD":6,
    "DEFENSA CIVIL":7,
    "PARTICIPACIÓN VECINAL":8
}

# Recorrer las filas del dataset y reemplazar las categorías con valores numéricos
for index, row in data.iterrows():
    categoria = row["USO_DEPENDENCIA"]
    if categoria in categorias:
        data.at[index, "USO_DEPENDENCIA"] = categorias[categoria]

# Guardar el resultado en un nuevo archivo CSV
data.to_csv('dataset2_numerico.csv', index=False)