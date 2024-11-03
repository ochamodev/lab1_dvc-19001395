import pandas as pd
import sys
import yaml
from sklearn.model_selection import train_test_split

def preprocess(input_file, train_output_file, test_output_file, features, target, test_size, random_state):
    # Cargar el dataset
    df = pd.read_csv(input_file)

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Seleccionar solo las columnas necesarias
    columns = features + [target]
    df = df[columns]

    # Dividir el dataset en entrenamiento y prueba
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Guardar los conjuntos de entrenamiento y prueba
    train_df = X_train.copy()
    train_df[target] = y_train
    test_df = X_test.copy()
    test_df[target] = y_test

    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)
    print(f"Preprocesamiento completado. Datos de entrenamiento guardados en {train_output_file} y datos de prueba en {test_output_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivos de salida para entrenamiento y prueba, archivo de parámetros
    input_file = sys.argv[1]
    train_output_file = sys.argv[2]
    test_output_file = sys.argv[3]
    params_file = sys.argv[4]

    # Leer parámetros desde params.yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    test_size = params['train']['test_size']
    random_state = params['train']['random_state']

    preprocess(input_file, train_output_file, test_output_file, features, target, test_size, random_state)
