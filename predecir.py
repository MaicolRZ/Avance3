import pandas as pd
import pickle

# Cargar el modelo desde el archivo
with open('modelo_regresion.pkl', 'rb') as archivo:
    model = pickle.load(archivo)

# Introducir los datos para realizar la predicción
mes = int(input("Introduce el mes (en formato numérico): "))
tipo_gasolina = input("Introduce el tipo de gasolina: ")
cantidad_galones = float(input("Introduce la cantidad de galones: "))

# Crear un DataFrame con los datos de entrada
nuevos_datos = pd.DataFrame({'MES_GASTO': [0], 'USO_DEPENDENCIA': [0], 'PERIODO': [0], 'TIPO_COMBUSTIBLE': [tipo_gasolina], 'PRECIO_GALON': [0], 'CANTIDAD_GALONES': [cantidad_galones], 'MONTO_CONSUMO': [0]})

# Codificar variables categóricas utilizando one-hot encoding
nuevos_datos_encoded = pd.get_dummies(nuevos_datos, drop_first=True)

# Asegurarse de que los nuevos datos tengan las mismas columnas que el modelo
columnas_modelo = list(nuevos_datos_encoded.columns)  # Obtener las columnas del DataFrame

# Agregar las columnas faltantes con valor 0
columnas_faltantes = set(columnas_modelo) - set(nuevos_datos_encoded.columns)
for columna in columnas_faltantes:
    nuevos_datos_encoded[columna] = 0

# Reordenar las columnas en el mismo orden que el modelo
nuevos_datos_encoded = nuevos_datos_encoded[columnas_modelo]

# Asignar los valores introducidos al nuevo DataFrame
nuevos_datos_encoded['MES_GASTO'] = mes
nuevos_datos_encoded['CANTIDAD_GALONES'] = cantidad_galones

# Realizar la predicción con el modelo cargado
prediccion = model.predict(nuevos_datos_encoded)

# Mostrar la predicción
print("El monto estimado gastado para el mes, tipo de gasolina y cantidad de galones ingresados es:", prediccion)


