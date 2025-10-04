# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metricas.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metricas.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metricas.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metricas.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Importación de librerías necesarias para el análisis de datos y machine learning
import pandas as pd  # Para manipulación y análisis de datos estructurados
import os  # Para operaciones del sistema operativo como manejo de rutas
import gzip  # Para compresión y descompresión de archivos
import pickle  # Para serialización de objetos Python (guardar modelos)
import json  # Para manejo de archivos JSON (guardar métricas)
from sklearn.model_selection import GridSearchCV  # Para búsqueda de hiperparámetros óptimos
from sklearn.ensemble import RandomForestClassifier  # Algoritmo de bosques aleatorios para clasificación
from sklearn.pipeline import Pipeline  # Para crear flujos de procesamiento de datos
from sklearn.compose import ColumnTransformer  # Para aplicar transformaciones a columnas específicas
from sklearn.preprocessing import OneHotEncoder  # Para codificación one-hot de variables categóricas
from sklearn.metricas import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix  # Métricas de evaluación

# Función para cargar datos desde archivos comprimidos en formato ZIP
# Esta función es fundamental para leer los datasets de entrenamiento y prueba
def carga(path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV comprimido en formato ZIP y lo convierte en un DataFrame de pandas.
    
    Args:
        path (str): Ruta al archivo ZIP que contiene el CSV
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados desde el archivo
    """
    # Lee el archivo CSV directamente desde el ZIP sin necesidad de descomprimir manualmente
    # index_col=False evita que pandas use la primera columna como índice
    # compression="zip" especifica que el archivo está comprimido en formato ZIP
    return pd.read_csv(path, index_col=False, compression="zip")

# Función para limpiar y preprocesar los datos según los requerimientos del problema
def limpieza(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza y preprocesamiento de los datos siguiendo las especificaciones:
    1. Renombra la columna objetivo
    2. Elimina columnas innecesarias
    3. Filtra registros con información no disponible
    4. Agrupa categorías de educación superiores a 4 en "others"
    
    Args:
        df (pd.DataFrame): DataFrame original con los datos sin procesar
    
    Returns:
        pd.DataFrame: DataFrame limpio y procesado
    """
    # Paso 1: Renombrar la columna objetivo para que sea más manejable
    # "default payment next month" -> "default"
    df = df.rename(columns={"default payment next month": "default"})
    
    # Paso 2: Eliminar la columna ID ya que no aporta información predictiva
    # La columna ID es solo un identificador único sin valor para el modelo
    df = df.drop(columns=["ID"])
    
    # Paso 3: Filtrar registros donde MARRIAGE = 0 (información no disponible)
    # Solo mantener registros con información válida de estado civil
    df = df.loc[df["MARRIAGE"] != 0] 
    
    # Paso 4: Filtrar registros donde EDUCATION = 0 (información no disponible)
    # Solo mantener registros con información válida de educación
    df = df.loc[df["EDUCATION"] != 0] 
    
    # Paso 5: Agrupar niveles de educación superiores a 4 en la categoría "others" (valor 4)
    # Esto simplifica las categorías de educación y reduce la dimensionalidad
    # Si EDUCATION < 4, mantener el valor original; si >= 4, asignar 4 (others)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    
    # Retornar el DataFrame limpio y procesado
    return df

# Función para crear el pipeline de procesamiento y clasificación
def crearPipe() -> Pipeline:
    """
    Crea un pipeline que incluye:
    1. Preprocesamiento de variables categóricas con One-Hot Encoding
    2. Clasificador de Random Forest
    
    Returns:
        Pipeline: Pipeline completo para entrenamiento y predicción
    """
    # Definir las variables categóricas que necesitan transformación
    # Estas variables serán codificadas usando One-Hot Encoding
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Crear el preprocesador que aplicará transformaciones específicas por tipo de variable
    preprocessor = ColumnTransformer(
        transformers=[
            # Transformador para variables categóricas
            # "cat": nombre del transformador
            # OneHotEncoder(): convierte variables categóricas en variables dummy/binarias
            # handle_unknown="ignore": maneja categorías no vistas durante el entrenamiento
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ],
        # remainder="passthrough": mantiene las variables numéricas sin cambios
        remainder="passthrough",
    )
    
    # Crear el pipeline completo que combina preprocesamiento y clasificación
    return Pipeline(
        steps=[
            # Paso 1: Preprocesamiento de los datos
            ("preprocessor", preprocessor),
            # Paso 2: Clasificación usando Random Forest
            # random_state=42: semilla para reproducibilidad de resultados
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

# Función para crear el estimador con optimización de hiperparámetros
def crearEst(pipeline: Pipeline) -> GridSearchCV:
    """
    Crea un estimador con GridSearchCV para encontrar los mejores hiperparámetros
    del Random Forest usando validación cruzada.
    
    Args:
        pipeline (Pipeline): Pipeline base para optimizar
    
    Returns:
        GridSearchCV: Estimador con búsqueda de hiperparámetros configurada
    """
    # Definir el espacio de búsqueda de hiperparámetros para Random Forest
    # Cada parámetro se prueba en combinación con todos los demás
    param_grid = {
        # Número de árboles en el bosque
        # Más árboles = mejor rendimiento pero mayor tiempo de entrenamiento
        "classifier__n_estimadors": [50, 100, 200],
        
        # Profundidad máxima de cada árbol
        # None = sin límite, valores numéricos = límite específico
        # Mayor profundidad puede llevar a overfitting
        "classifier__max_depth": [None, 5, 10, 20],
        
        # Número mínimo de muestras requeridas para dividir un nodo interno
        # Valores más altos previenen overfitting
        "classifier__min_samples_split": [2, 5, 10],
        
        # Número mínimo de muestras requeridas en un nodo hoja
        # Valores más altos suavizan el modelo
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    # Crear el objeto GridSearchCV para búsqueda exhaustiva de hiperparámetros
    return GridSearchCV(
        pipeline,  # Pipeline base a optimizar
        param_grid,  # Espacio de búsqueda de hiperparámetros
        cv=10,  # Validación cruzada con 10 folds (requisito del problema)
        scoring="balanced_accuracy",  # Métrica de evaluación (precisión balanceada)
        n_jobs=-1,  # Usar todos los procesadores disponibles para paralelización
        verbose=2,  # Nivel de detalle en la salida (mostrar progreso)
        refit=True,  # Reentrenar el modelo con los mejores parámetros en todo el dataset
    )

# Función para guardar el modelo entrenado de forma comprimida
def guardar(path: str, estimador: GridSearchCV):
    """
    Guarda el modelo entrenado en un archivo comprimido usando gzip.
    Esto reduce significativamente el tamaño del archivo.
    
    Args:
        path (str): Ruta donde guardar el modelo
        estimador (GridSearchCV): Modelo entrenado a guardar
    """
    # Crear el directorio destino si no existe
    # exist_ok=True evita errores si el directorio ya existe
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    
    # Guardar el modelo usando compresión gzip
    # "wb": modo binario de escritura necesario para pickle
    with gzip.open(path, "wb") as f:
        # pickle.dump() serializa el objeto Python y lo guarda en el archivo
        # Esto permite cargar el modelo más tarde para hacer predicciones
        pickle.dump(estimador, f)

# Función para calcular las métricas de evaluación del modelo
def metricas(dataset_name: str, y_true, y_pred) -> dict:
    """
    Calcula las métricas principales de evaluación para un modelo de clasificación binaria.
    
    Args:
        dataset_name (str): Nombre del dataset ("train" o "test")
        y_true: Valores reales de la variable objetivo
        y_pred: Valores predichos por el modelo
    
    Returns:
        dict: Diccionario con las métricas calculadas
    """
    return {
        "type": "metrics",  # Tipo de registro para identificar en el archivo JSON
        "dataset": dataset_name,  # Identificador del dataset (entrenamiento o prueba)
        
        # Precision: TP / (TP + FP) - Proporción de predicciones positivas que son correctas
        # zero_division=0: retorna 0 si no hay predicciones positivas
        "precision": precision_score(y_true, y_pred, zero_division=0),
        
        # Balanced Accuracy: (Sensitivity + Specificity) / 2
        # Es útil cuando las clases están desbalanceadas
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        
        # Recall (Sensitivity): TP / (TP + FN) - Proporción de casos positivos detectados
        # zero_division=0: retorna 0 si no hay casos positivos reales
        "recall": recall_score(y_true, y_pred, zero_division=0),
        
        # F1-Score: Media armónica entre precision y recall
        # Balanceada entre precision y recall
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

# Función para calcular y estructurar la matriz de confusión
def metricasConfusion(dataset_name: str, y_true, y_pred) -> dict:
    """
    Calcula la matriz de confusión y la estructura en el formato requerido.
    La matriz de confusión muestra los aciertos y errores del modelo de forma detallada.
    
    Args:
        dataset_name (str): Nombre del dataset ("train" o "test")
        y_true: Valores reales de la variable objetivo
        y_pred: Valores predichos por el modelo
    
    Returns:
        dict: Diccionario con la matriz de confusión estructurada
    """
    # Calcular la matriz de confusión
    # cm[0][0]: Verdaderos Negativos (TN) - Predijo 0 y era 0
    # cm[0][1]: Falsos Positivos (FP) - Predijo 1 y era 0
    # cm[1][0]: Falsos Negativos (FN) - Predijo 0 y era 1
    # cm[1][1]: Verdaderos Positivos (TP) - Predijo 1 y era 1
    cm = confusion_matrix(y_true, y_pred)
    
    # Estructurar la matriz en el formato requerido
    return {
        "type": "cm_matrix",  # Tipo de registro para identificar en el archivo JSON
        "dataset": dataset_name,  # Identificador del dataset
        
        # Casos donde el valor real es 0 (no default)
        "true_0": {
            "predicted_0": int(cm[0][0]),  # Verdaderos Negativos
            "predicted_1": int(cm[0][1])   # Falsos Positivos
        },
        
        # Casos donde el valor real es 1 (default)
        "true_1": {
            "predicted_0": int(cm[1][0]),  # Falsos Negativos
            "predicted_1": int(cm[1][1])   # Verdaderos Positivos
        },
    }

# Función principal que ejecuta todo el flujo de trabajo del machine learning
def main():
    """
    Función principal que coordina todo el proceso de machine learning:
    1. Carga y limpieza de datos
    2. Separación de variables predictoras y objetivo
    3. Creación y entrenamiento del modelo
    4. Evaluación y guardado de resultados
    """
    
    # PASO 1: CARGA DE DATOS
    # Cargar los datos de prueba desde el archivo comprimido
    # Estos datos se usarán para evaluar el rendimiento final del modelo
    test_df = carga(os.path.join("files/input/", "test_data.csv.zip"))
    
    # Cargar los datos de entrenamiento desde el archivo comprimido
    # Estos datos se usarán para entrenar y optimizar el modelo
    train_df = carga(os.path.join("files/input/", "train_data.csv.zip"))

    # PASO 2: LIMPIEZA Y PREPROCESAMIENTO
    # Aplicar las transformaciones de limpieza al conjunto de prueba
    # Esto incluye renombrar columnas, eliminar registros inválidos, etc.
    test_df = limpieza(test_df)
    
    # Aplicar las mismas transformaciones al conjunto de entrenamiento
    # Es crucial aplicar las mismas transformaciones a ambos conjuntos
    train_df = limpieza(train_df)

    # PASO 3: SEPARACIÓN DE VARIABLES PREDICTORAS Y OBJETIVO
    # Separar las variables predictoras (X) del conjunto de prueba
    # Se excluye la columna "default" que es la variable objetivo
    x_test = test_df.drop(columns=["default"])
    
    # Extraer la variable objetivo (y) del conjunto de prueba
    # Esta es la variable que queremos predecir
    y_test = test_df["default"]

    # Separar las variables predictoras (X) del conjunto de entrenamiento
    # Estas son todas las variables que usaremos para hacer predicciones
    x_train = train_df.drop(columns=["default"])
    
    # Extraer la variable objetivo (y) del conjunto de entrenamiento
    # Estos son los valores reales que el modelo aprenderá a predecir
    y_train = train_df["default"]

    # PASO 4: CREACIÓN Y CONFIGURACIÓN DEL MODELO
    # Crear el pipeline de procesamiento y clasificación
    # Esto incluye la codificación de variables categóricas y el algoritmo Random Forest
    pipeline = crearPipe()

    # Crear el estimador con optimización de hiperparámetros usando GridSearchCV
    # Esto buscará automáticamente los mejores parámetros para el modelo
    estimador = crearEst(pipeline)
    
    # PASO 5: ENTRENAMIENTO DEL MODELO
    # Entrenar el modelo con los datos de entrenamiento
    # GridSearchCV probará todas las combinaciones de hiperparámetros
    # y seleccionará la mejor usando validación cruzada
    estimador.fit(x_train, y_train)

    # PASO 6: GUARDADO DEL MODELO
    # Guardar el modelo entrenado en formato comprimido
    # Esto permite reutilizar el modelo más tarde sin necesidad de reentrenarlo
    guardar(os.path.join("files/models/", "model.pkl.gz"), estimador)

    # PASO 7: EVALUACIÓN DEL MODELO
    # Realizar predicciones sobre el conjunto de prueba
    # Estos son datos que el modelo nunca ha visto durante el entrenamiento
    y_test_pred = estimador.predict(x_test)
    
    # Calcular métricas de evaluación para el conjunto de prueba
    # Estas métricas indican qué tan bien generaliza el modelo
    precision_test = metricas("test", y_test, y_test_pred)
    
    # Realizar predicciones sobre el conjunto de entrenamiento
    # Esto permite evaluar si el modelo está sobreajustado (overfitting)
    y_train_pred = estimador.predict(x_train)
    
    # Calcular métricas de evaluación para el conjunto de entrenamiento
    # Comparar con las métricas de prueba para detectar overfitting
    precision_train = metricas("train", y_train, y_train_pred)

    # PASO 8: CÁLCULO DE MATRICES DE CONFUSIÓN
    # Calcular la matriz de confusión para el conjunto de prueba
    # Esto muestra detalladamente los tipos de errores del modelo
    conf_test = metricasConfusion("test", y_test, y_test_pred)
    
    # Calcular la matriz de confusión para el conjunto de entrenamiento
    # Útil para comparar el rendimiento entre entrenamiento y prueba
    conf_train = metricasConfusion("train", y_train, y_train_pred)

    # PASO 9: GUARDADO DE RESULTADOS
    # Crear el directorio de salida si no existe
    os.makedirs("../files/output/", exist_ok=True)
    
    # Guardar todas las métricas en un archivo JSON
    # Cada línea del archivo contiene un diccionario con métricas específicas
    with open(os.path.join("../files/output/", "metricas.json"), "w") as file:
        # Escribir métricas de entrenamiento (precision, recall, f1-score, etc.)
        file.write(json.dumps(precision_train) + "\n")
        
        # Escribir métricas de prueba (precision, recall, f1-score, etc.)
        file.write(json.dumps(precision_test) + "\n")
        
        # Escribir matriz de confusión del conjunto de entrenamiento
        file.write(json.dumps(conf_train) + "\n")
        
        # Escribir matriz de confusión del conjunto de prueba
        file.write(json.dumps(conf_test) + "\n")

# Punto de entrada del programa
# Se ejecuta solo cuando el archivo se ejecuta directamente (no cuando se importa)
if __name__ == "__main__":
    main()