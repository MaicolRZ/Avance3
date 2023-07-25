import pandas as pd
import pickle

# Cargar el modelo desde el archivo
with open('modelo_regresion.pkl', 'rb') as archivo:
    model = pickle.load(archivo)

# Introducir los valores para realizar la predicción
uso_dependencia = 1
periodo = 1
tipo_combustible = 1
precio_galon = 3.5
cantidad_galones = 10
monto_consumo = 35

# Crear un DataFrame con los valores de entrada
nuevos_datos = pd.DataFrame({'USO_DEPENDENCIA': [uso_dependencia],
                             'PERIODO': [periodo],
                             'TIPO_COMBUSTIBLE': [tipo_combustible],
                             'PRECIO_GALON': [precio_galon],
                             'CANTIDAD_GALONES': [cantidad_galones],
                             'MONTO_CONSUMO': [monto_consumo]})

# Codificar variables categóricas utilizando one-hot encoding
nuevos_datos_encoded = pd.get_dummies(nuevos_datos, drop_first=True)

# Obtener las columnas del modelo entrenado
columnas_modelo = list(nuevos_datos_encoded.columns)

# Verificar si hay columnas faltantes
columnas_faltantes = set(columnas_modelo) - set(nuevos_datos_encoded.columns)

# Agregar las columnas faltantes con valor 0
for columna in columnas_faltantes:
    nuevos_datos_encoded[columna] = 0

# Reordenar las columnas en el mismo orden que el modelo
nuevos_datos_encoded = nuevos_datos_encoded[columnas_modelo]

# Realizar la predicción con el modelo cargado
prediccion = model.predict(nuevos_datos_encoded)

# Mostrar la predicción
print("La predicción del gasto para los valores ingresados es:", prediccion)