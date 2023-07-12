import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import joblib

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv("dataset.csv", sep=";", encoding="latin-1")

# Mostrar los primeros registros del conjunto de datos
print(data.head())

# Obtener la forma (dimensiones) del conjunto de datos
print(data.shape)

# Crear una copia del conjunto de datos original
data_t = data

# Calcular la proporción de valores nulos para cada columna
print(data_t.isna().sum() / len(data_t))

# Eliminar filas con valores nulos
data_t = data_t.dropna()

# Obtener la nueva forma del conjunto de datos después de eliminar filas con valores nulos
print(data_t.shape)

# Eliminar filas duplicadas en el conjunto de datos
data_t = data_t.drop_duplicates()

# Obtener la forma final del conjunto de datos después de eliminar filas duplicadas
print(data_t.shape)

# Contar los valores únicos en la columna 'MES_GASTO'
print(pd.value_counts(data['MES_GASTO']))

# Contar los valores únicos en la columna 'TIPO_COMBUSTIBLE'
print(pd.value_counts(data['TIPO_COMBUSTIBLE']))

# Convertir columnas categóricas en variables ficticias (one-hot encoding)
data_t = pd.get_dummies(data_t, columns=['MES_GASTO', 'TIPO_COMBUSTIBLE'])

# Obtener la forma del conjunto de datos después de aplicar one-hot encoding
print(data_t.shape)

# Mostrar los primeros registros del conjunto de datos después de aplicar las transformaciones
print(data_t.head())

# Separar la variable objetivo (PRECIO_GALON) del conjunto de datos
Y = data_t['PRECIO_GALON']
X = data_t.drop(['PRECIO_GALON'], axis=1)

# Mostrar los primeros registros de las variables predictoras y la variable objetivo
print(X.head())
print(Y.head())

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Crear un modelo de regresión lineal
modelo_regresion = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
modelo_regresion.fit(X_train, Y_train)

# Obtener las predicciones del modelo sobre el conjunto de entrenamiento
y_pred = modelo_regresion.predict(X_train)

# Calcular las métricas de evaluación del modelo sobre el conjunto de entrenamiento
print("MSE: %.2f" % mean_squared_error(Y_train, y_pred, squared=True))
print("RMSE: %.2f" % mean_squared_error(Y_train, y_pred, squared=False))
print("MAE: %.2f" % mean_absolute_error(Y_train, y_pred))
print("R2: %.2f" % r2_score(Y_train, y_pred))

# Obtener las predicciones del modelo sobre el conjunto de prueba
y_pred = modelo_regresion.predict(X_test)

# Calcular las métricas de evaluación del modelo sobre el conjunto de prueba
print("MSE: %.2f" % mean_squared_error(Y_test, y_pred, squared=True))
print("RMSE: %.2f" % mean_squared_error(Y_test, y_pred, squared=False))
print("MAE: %.2f" % mean_absolute_error(Y_test, y_pred))
print("R2: %.2f" % r2_score(Y_test, y_pred))

# Ajustar el modelo a todos los datos
modelo_regresion.fit(X, Y)

# Obtener los coeficientes de regresión y la intersección
print(modelo_regresion.coef_)
print(modelo_regresion.intercept_)

# Guardar el modelo entrenado en un archivo
joblib.dump(modelo_regresion, "ModeloRegresion.joblib")
