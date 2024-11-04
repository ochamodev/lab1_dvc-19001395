import pandas as pd
import joblib
import yaml
import sys
from sklearn.metrics import mean_squared_error, r2_score
import csv

def calculate_metrics(model, X_test, y_test):
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "MSE": mse,
        "R2 Score": r2
    }

def get_feature_importances(model, features):
    # Obtener la importancia de características si el modelo lo permite
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return feature_importance
    else:
        return None

def main(test_file, params_file, model_file, output_file):
    # Cargar el archivo de prueba y los parámetros
    test_df = pd.read_csv(test_file)
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X_test = test_df[features]
    y_test = test_df[target]

    # Cargar el modelo
    model = joblib.load(model_file)

    # Calcular métricas
    metrics = calculate_metrics(model, X_test, y_test)
    print("Métricas de rendimiento:", metrics)

    # Obtener importancia de características
    feature_importances = get_feature_importances(model, features)
    if feature_importances:
        print("Importancia de características:")
        for feature, importance in feature_importances:
            print(f"{feature}: {importance}")

    # Guardar resultados en CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Métrica", "Valor"])
        for metric, value in metrics.items():
            writer.writerow([metric, value])

        # Agregar parámetros finales del modelo
        writer.writerow([])
        writer.writerow(["Parámetros del modelo"])
        for param, value in model.get_params().items():
            writer.writerow([param, value])

        # Agregar importancia de características
        if feature_importances:
            writer.writerow([])
            writer.writerow(["Característica", "Importancia"])
            for feature, importance in feature_importances:
                writer.writerow([feature, importance])

    print(f"Resultados guardados en {output_file}")

if __name__ == "__main__":
    # Argumentos: archivo de prueba, archivo de parámetros, archivo del modelo, archivo de salida
    test_file = sys.argv[1]
    params_file = sys.argv[2]
    model_file = sys.argv[3]
    output_file = sys.argv[4]

    main(test_file, params_file, model_file, output_file)
