import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
import joblib

# Title of the app
st.title("Avance Trabajo 3")

# Header
st.header("Esta es una aplicación de Streamlit con Machine Learning")

# Leer el archivo CSV existente o crear uno nuevo si no existe
try:
    df = pd.read_csv("dataset2.csv", sep=";", encoding="latin-1")
except FileNotFoundError:
    df = pd.DataFrame()

# Registrar datos en el archivo CSV
if st.button("Registrar"):
    with st.form("form_registrar"):
        # Crear inputs para los títulos
        codigo_entidad = "301285"
        codigo_ubigeo = "150136"
        codigo_pais = "PE"
        departamento = "Lima"
        provincia = "Lima"
        distrito = "San Miguel"
        nombre_uo = "Unidad de Logística y Control Patrimonial"
        gobierno_local = "San Miguel"
        ruc_gobierno_local = "20131372184"
        ano_gasto = "2021"
        mes_options = list(range(1, 13))
        mes_gasto = st.selectbox("MES_GASTO", mes_options)
        usodepen1 = ["SERENAZGO","LIMPIEZA PUBLICA","PARQUES Y JARDINES","TRANSITO","MANTENIMIENTO","GERENCIA DE SALUD","DEFENSA CIVIL","PARTICIPACIÓN VECINAL"]
        uso_dependencia = st.selectbox("USO_DEPENDENCIA",usodepen1)
        nro_placa_cod_vehiculo = st.text_input("NRO_PLACA_COD_VEHICULO")
        numero_vale = "69983"
        fecha_consumo = st.date_input("FECHA_CONSUMO")
        period=["1er QUINCENA","2da QUINCENA"]
        periodo = st.selectbox("PERIODO",period)
        tipocomb=["DIESEL","GASOHOL 90","GASOHOL 97"]
        tipocombustible =  st.selectbox("TIPOCOMBUSTIBLE",tipocomb)
        preciogalon = st.text_input("PRECIOGALON")
        cantidadgalones = st.text_input("CANTIDADGALONES")

        submitted = st.form_submit_button("Guardar")

    # Verificar si el formulario ha sido enviado y guardar los datos
    if submitted:
        # Crear un diccionario con los datos ingresados
        datos = {
            "CODIGO_ENTIDAD": codigo_entidad,
            "CODIGO_UBIGEO": codigo_ubigeo,
            "CODIGO_PAIS": codigo_pais,
            "DEPARTAMENTO": departamento,
            "PROVINCIA": provincia,
            "DISTRITO": distrito,
            "NOMBRE_UO": nombre_uo,
            "GOBIERNO_LOCAL": gobierno_local,
            "RUC_GOBIERNO_LOCAL": ruc_gobierno_local,
            "AÑO_GASTO": ano_gasto,
            "MES_GASTO": mes_gasto,
            "USO_DEPENDENCIA": uso_dependencia,
            "NRO_PLACA_COD_VEHICULO": nro_placa_cod_vehiculo,
            "NUMERO_VALE": numero_vale,
            "FECHA_CONSUMO": fecha_consumo,
            "PERIODO": periodo,
            "TIPOCOMBUSTIBLE": tipocombustible,
            "PRECIOGALON": preciogalon,
            "CANTIDADGALONES": cantidadgalones
        }

        # Agregar los datos al DataFrame
        df = df.append(datos, ignore_index=True)

        # Guardar el DataFrame en el archivo CSV
        df.to_csv("dataset2.csv", index=False, sep=";", encoding="latin-1")

        st.success("Datos guardados correctamente en el archivo dataset.csv")

# Función para guardar el DataFrame en el archivo CSV

# Mostrar gráfico de dispersión
st.subheader("Gráfico de dispersión")
x_column = st.selectbox("Seleccionar columna para el eje x", df.columns)
y_column = st.selectbox("Seleccionar columna para el eje y", df.columns)
plt.figure(figsize=(8, 6))
plt.scatter(df[x_column], df[y_column])
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.xticks(rotation=45)
st.pyplot(plt)

# Mostrar histograma
st.subheader("Histograma")
column_options1 = ["PERIODO", "USO_DEPENDENCIA","TIPOCOMBUSTIBLE", "PRECIOGALON","CANTIDADGALONES"]
column = st.selectbox("Seleccionar columna para el histograma", column_options1)
plt.figure(figsize=(15, 6))
sns.histplot(df[column], kde=True)
plt.xlabel(column)
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
st.pyplot(plt)

# Mostrar gráfico de barras

st.subheader("Gráfico de barras")
column_options2 = ["PERIODO", "USO_DEPENDENCIA","TIPOCOMBUSTIBLE", "PRECIOGALON"]
column = st.selectbox("Seleccionar columna para el gráfico de barras", column_options2)
counts = df[column].value_counts()
plt.figure(figsize=(15, 6))
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel(column)
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
st.pyplot(plt)

# Agregar predicción de regresión lineal
st.subheader("Predicción de regresión lineal")

# Obtener las columnas seleccionadas para la regresión lineal
feature_column = st.selectbox("Seleccionar columna para la característica (X)", ["CANTIDADGALONES"], key="feature_column")
target_column = st.selectbox("Seleccionar columna para el objetivo (y)", ["MONTOCONSUMO"], key="target_column")

# Imputar los valores faltantes en las columnas seleccionadas
imputer = SimpleImputer(strategy="mean")
imputed_df = pd.DataFrame(imputer.fit_transform(df[[feature_column, target_column]]), columns=[feature_column, target_column])

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos imputados
model.fit(imputed_df[[feature_column]], imputed_df[target_column])

# Mostrar coeficientes e intercepto
st.write("Coeficiente:", model.coef_[0])
st.write("Intercepto:", model.intercept_)

# Hacer una predicción
prediction_input = st.number_input("Ingresar valor para la característica (X) a predecir", value=0, key="prediction_input")
prediction_output = model.predict([[prediction_input]])

st.write("Predicción:", prediction_output[0])

# ...

# Cargar el modelo entrenado
modelo_regresion = joblib.load("ModeloRegresion.joblib")

# Obtener los valores de entrada del usuario
mes_gasto = float(st.number_input("Ingrese el valor de MES_GASTO: "))
uso_dependencia = int(st.number_input("Ingrese el valor de USO_DEPENDENCIA: "))
periodo = int(st.number_input("Ingrese el valor de PERIODO: "))
tipo_combustible = int(st.number_input("Ingrese el valor de TIPO_COMBUSTIBLE: "))
cantidad_galones = float(st.number_input("Ingrese el valor de CANTIDAD_GALONES: "))
monto_consumo = float(st.number_input("Ingrese el valor de MONTO_CONSUMO: "))

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
st.write("La predicción es:", prediccion)