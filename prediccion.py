import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo_regresion = joblib.load("ModeloRegresion.joblib")

# Obtener los valores de entrada del usuario
mes_gasto = float(input("Ingrese el valor de MES_GASTO: "))
uso_dependencia = int(input("Ingrese el valor de USO_DEPENDENCIA: "))
periodo = int(input("Ingrese el valor de PERIODO: "))
tipo_combustible = int(input("Ingrese el valor de TIPO_COMBUSTIBLE: "))
cantidad_galones = float(input("Ingrese el valor de CANTIDAD_GALONES: "))
monto_consumo = float(input("Ingrese el valor de MONTO_CONSUMO: "))

# Crear un DataFrame con los valores de entrada
nuevos_datos = pd.DataFrame({
    'MES_GASTO': [mes_gasto],
    'USO_DEPENDENCIA': [uso_dependencia],
    'PERIODO': [periodo],
    'TIPO_COMBUSTIBLE': [tipo_combustible],
    'CANTIDAD_GALONES': [cantidad_galones],
    'MONTO_CONSUMO': [monto_consumo]
})

# Cargar el archivo CSV original utilizado durante el entrenamiento para obtener las columnas dummy
datos_entrenamiento = pd.read_csv("dataset.csv", sep=";", encoding="latin-1")

# Aplicar la codificación one-hot a las columnas categóricas en los nuevos datos
columnas_categoricas = ['TIPO_COMBUSTIBLE']
nuevos_datos_codificados = pd.get_dummies(nuevos_datos, columns=columnas_categoricas)

# Asegurarse de que las columnas codificadas coincidan con las columnas del conjunto de entrenamiento
columnas_faltantes = set(datos_entrenamiento.columns) - set(nuevos_datos_codificados.columns)
for columna in columnas_faltantes:
    nuevos_datos_codificados[columna] = 0

# Reordenar las columnas para que coincidan con el conjunto de entrenamiento
nuevos_datos_codificados = nuevos_datos_codificados[datos_entrenamiento.columns]

# Realizar la predicción en los nuevos datos codificados utilizando el modelo cargado
prediccion = modelo_regresion.predict(nuevos_datos_codificados)

# Mostrar la predicción
print("La predicción es:", prediccion)
