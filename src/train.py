import pandas as pd
import joblib
import yaml
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model_params):
    # Filtrar el parámetro "type" que no es necesario para la creación del modelo
    model_params = {k: v for k, v in model_params.items() if k != "type"}
    
    # Crear una instancia del modelo basado en el nombre
    if model_name == "linear_regression":
        model = Ridge(**model_params)
    elif model_name == "random_forest":
        model = RandomForestRegressor(**model_params)
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(**model_params)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

def main(train_file, test_file, params_file, output_model_file):
    # Cargar los datos de entrenamiento y prueba
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Leer parámetros desde params.yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    best_model = None
    best_mse = float("inf")
    best_model_name = None

    # Iterar sobre los modelos definidos en params.yaml
    for model_name, model_params in params["models"].items():
        # Entrenar y evaluar el modelo
        model, mse = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model_params)

        print(f"{model_name} - MSE: {mse}")

        # Actualizar el mejor modelo si tiene el menor MSE
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = model_name

    # Guardar el mejor modelo en un archivo
    joblib.dump(best_model, output_model_file)
    print(f"El mejor modelo es {best_model_name} con MSE: {best_mse}. Guardado en {output_model_file}")

if __name__ == "__main__":
    import sys
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    params_file = sys.argv[3]
    output_model_file = sys.argv[4]

    main(train_file, test_file, params_file, output_model_file)
