# ============================================================
# APP STREAMLIT PARA PREDICCIÓN DE DIABETES
# Modelo: Random Forest creado en Orange Data Mining
# ============================================================

# Importamos Streamlit para crear la aplicación web
import streamlit as st

# Importamos pickle para cargar el modelo guardado desde Orange
import pickle

# Importamos numpy para crear arreglos numéricos
import numpy as np

# Importamos Table porque Orange necesita recibir los datos
# en formato de tabla de Orange
from Orange.data import Table

# Importamos Model para indicar qué queremos obtener:
# la predicción y las probabilidades
from Orange.classification import Model


# ============================================================
# CONFIGURACIÓN INICIAL DE LA PÁGINA
# ============================================================

# Configura el título de la pestaña del navegador,
# el ícono y el ancho de la app
st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# TÍTULO E IMAGEN DE LA APLICACIÓN
# ============================================================

# Título principal que verá el usuario
st.title("🩺 Predicción de Diabetes")

# Texto introductorio de la app
st.write(
    "Aplicación creada con Streamlit usando un modelo Random Forest "
    "generado en Orange Data Mining."
)


# ============================================================
# CARGA DEL MODELO ENTRENADO
# ============================================================

# Esta función carga el modelo guardado en formato .pkcls.
# El decorador @st.cache_resource permite que el modelo se cargue
# una sola vez y no cada vez que el usuario interactúa con la app.
@st.cache_resource
def cargar_modelo():
    with open("model_rf_diabetes.pkcls", "rb") as archivo:
        modelo = pickle.load(archivo)
    return modelo


# Llamamos a la función para cargar el modelo
modelo = cargar_modelo()


# ============================================================
# OBTENER INFORMACIÓN DEL MODELO
# ============================================================

# En Orange, el modelo conserva información sobre:
# - los atributos usados para entrenar
# - la variable objetivo
# - posibles columnas meta
clase = modelo.domain.class_var
metas = modelo.domain.metas


# ============================================================
# FORMULARIO DE ENTRADA DE DATOS
# ============================================================

# Mensaje informativo para el usuario
st.info(
    "Ingrese los datos del paciente. El modelo utiliza solo 5 atributos: "
    "Embarazos, Glucosa, Insulina, IMC y Edad."
)

# Subtítulo del formulario
st.subheader("Datos del paciente")


# Cada number_input crea una caja numérica en la app.
# Estos valores deben corresponder a los atributos usados
# al entrenar el modelo en Orange.

embarazos = st.number_input(
    "Embarazos",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

glucosa = st.number_input(
    "Glucosa",
    min_value=0,
    max_value=250,
    value=85,
    step=1
)

insulina = st.number_input(
    "Insulina",
    min_value=0,
    max_value=900,
    value=80,
    step=1
)

imc = st.number_input(
    "IMC",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

edad = st.number_input(
    "Edad",
    min_value=1,
    max_value=120,
    value=25,
    step=1
)


# ============================================================
# BOTÓN PARA REALIZAR LA PREDICCIÓN
# ============================================================

# Cuando el usuario presiona el botón "Predecir",
# se ejecuta todo el bloque de código que está dentro del if.
if st.button("Predecir"):

    # --------------------------------------------------------
    # 1. Crear el arreglo con los datos ingresados
    # --------------------------------------------------------

    # El orden de las columnas es muy importante.
    # Debe ser el mismo orden usado por el modelo en Orange:
    # Embarazos, Glucosa, Insulina, IMC, Edad
    X = np.array(
        [[embarazos, glucosa, insulina, imc, edad]],
        dtype=float
    )


    # --------------------------------------------------------
    # 2. Crear la columna vacía para la variable objetivo
    # --------------------------------------------------------

    # Orange espera recibir también la columna de la clase.
    # Como estamos prediciendo, no conocemos la clase real.
    # Por eso enviamos np.nan, que significa "valor desconocido".
    Y = np.array([[np.nan]])


    # --------------------------------------------------------
    # 3. Crear columnas meta vacías si el modelo las necesita
    # --------------------------------------------------------

    # Algunos modelos de Orange conservan columnas meta.
    # Si existen, debemos enviarlas vacías para que la estructura
    # de la tabla sea compatible con el modelo.
    if len(metas) > 0:
        M = np.empty((1, len(metas)), dtype=object)
        M[:] = None
    else:
        M = None


    # --------------------------------------------------------
    # 4. Convertir los datos al formato de Orange
    # --------------------------------------------------------

    # El modelo fue creado en Orange, por eso no recibe directamente
    # un arreglo de numpy. Primero convertimos los datos a una tabla
    # compatible con Orange usando el dominio del modelo.
    tabla_orange = Table.from_numpy(
        modelo.domain,
        X,
        Y=Y,
        metas=M
    )


    # --------------------------------------------------------
    # 5. Realizar la predicción
    # --------------------------------------------------------

    # Model.Value devuelve la clase predicha.
    prediccion = modelo(tabla_orange, Model.Value)

    # Model.Probs devuelve las probabilidades de cada clase.
    probabilidades = modelo(tabla_orange, Model.Probs)


    # --------------------------------------------------------
    # 6. Interpretar la clase predicha
    # --------------------------------------------------------

    # Orange devuelve un índice interno de la clase.
    # Por eso usamos clase.values para obtener la etiqueta real.
    indice_clase = int(prediccion[0])
    etiqueta_predicha = clase.values[indice_clase]


    # --------------------------------------------------------
    # 7. Mostrar el resultado en pantalla
    # --------------------------------------------------------

    st.subheader("Resultado de la predicción")

    # En este dataset:
    # 0 representa No diabetes
    # 1 representa Posible diabetes
    if etiqueta_predicha == "0":
        st.success("Predicción: No diabetes")
    elif etiqueta_predicha == "1":
        st.error("Predicción: Posible diabetes")
    else:
        st.warning(f"Predicción: {etiqueta_predicha}")


    # --------------------------------------------------------
    # 8. Mostrar las probabilidades
    # --------------------------------------------------------

    st.subheader("Probabilidades")

    # Recorremos todas las clases del modelo y mostramos
    # la probabilidad correspondiente.
    for i, nombre_clase in enumerate(clase.values):
        if nombre_clase == "0":
            st.write(f"No diabetes: {probabilidades[0][i]:.2%}")
        elif nombre_clase == "1":
            st.write(f"Diabetes: {probabilidades[0][i]:.2%}")
        else:
            st.write(f"Clase {nombre_clase}: {probabilidades[0][i]:.2%}")


    # --------------------------------------------------------
    # 9. Mensaje de advertencia
    # --------------------------------------------------------

    # Es importante aclarar que la app es educativa
    # y no reemplaza una evaluación médica.
    st.warning(
        "Este resultado es una predicción generada por un modelo de machine learning "
        "y no reemplaza una evaluación médica profesional."
    )
