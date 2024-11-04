import pandas as pd
import joblib
import yaml
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def get_model_and_params(model_name, param_grid):
    if model_name == "linear_regression":
        model = Ridge()
        param_grid = {"alpha": param_grid.get("alpha", [1.0])}
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=param_grid.get("random_state", 42))
        param_grid = {
            "n_estimators": param_grid.get("n_estimators", [100]),
            "max_depth": param_grid.get("max_depth", [None])
        }
    elif model_name == "gradient_boost":
        model = GradientBoostingRegressor(random_state=param_grid.get("random_state", 42))
        param_grid = {
            "n_estimators": param_grid.get("n_estimators", [100]),
            "learning_rate": param_grid.get("learning_rate", [0.1])
        }
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    return model, param_grid

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

    # Iterar sobre los modelos y optimizar hiperparámetros
    for model_name, model_params in params["models_with_optimization_and_training"].items():
        model, param_grid = get_model_and_params(model_name, model_params)
        
        # Configurar Grid Search con validación cruzada
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)
        
        # Obtener el mejor modelo y evaluar
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        print(f"{model_name} - Mejor MSE: {mse} con parámetros: {grid_search.best_params_}")

        # Actualizar el mejor modelo si tiene el menor MSE
        if mse < best_mse:
            best_mse = mse
            best_model = best_estimator
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
