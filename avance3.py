import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

# Title of the app
st.title("Avance Trabajo 3")

# Header
st.header("Esta es una aplicación de Streamlit con Machine Learning")

# Leer el archivo CSV existente o crear uno nuevo si no existe
try:
    df = pd.read_csv("./machine/dataset.csv", sep=";", encoding="latin-1")
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
        fecha_corte = st.date_input("FECHA_CORTE")

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
            "CANTIDADGALONES": cantidadgalones,
            "FECHA_CORTE": fecha_corte
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

feature_column = st.selectbox("Seleccionar columna para la característica (X)", ["CANTIDADGALONES"])
target_column = st.selectbox("Seleccionar columna para el objetivo (y)", ["MONTOCONSUMO"])

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
prediction_input = st.number_input("Ingresar valor para Cantidad de Galones (X) a predecir", value=0)
prediction_output = model.predict([[prediction_input]])

st.write("Predicción:", prediction_output[0])

# Mostrar tabla con los datos del CSV
st.subheader("Tabla de datos del CSV")
st.dataframe(df)

# Crear el modelo de árbol de regresión
model_tree = DecisionTreeRegressor()

# Entrenar el modelo con los datos imputados
model_tree.fit(imputed_df[[feature_column]], imputed_df[target_column])

# Hacer una predicción con árbol de regresión
prediction_input_tree = st.number_input("Ingresar valor para Cantidad de Galones (X) a predecir (Árbol)", value=0)
prediction_output_tree = model_tree.predict([[prediction_input_tree]])

st.write("Predicción Árbol de Regresión:", prediction_output_tree[0])



#Regresion Lasso
# Agregar predicción de regresión Lasso
st.subheader("Predicción de regresión Lasso")

# Crear el modelo de regresión Lasso
model_lasso = Lasso()

# Entrenar el modelo con los datos imputados
model_lasso.fit(imputed_df[[feature_column]], imputed_df[target_column])

# Mostrar coeficientes e intercepto
st.write("Coeficientes:", model_lasso.coef_[0])
st.write("Intercepto:", model_lasso.intercept_)

# Hacer una predicción
prediction_input_lasso = st.number_input("Ingresar valor para Cantidad de Galones (X) a predecir (Lasso)", value=0)
prediction_output_lasso = model_lasso.predict([[prediction_input_lasso]])

st.write("Predicción Lasso:", prediction_output_lasso[0])