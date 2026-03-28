from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import time
from math import e
import pandas as pd

import mlflow

from phd_workflows.src.scripts.elastic_pattern import ElasticPattern

def predict_elastic_pattern(sample, elastic_patterns):
    res = -1
    weights = []

    for elastic_pattern in elastic_patterns:
        weights.append([elastic_pattern.compare(sample), elastic_pattern])

    # min_weight = 9999999999999999999999999
    min_weight_index = weights.index(min(weights))
    res = elastic_patterns[min_weight_index]

    return res

def test_elastic_patterns(X_test, y_test, elastic_patterns):

    sucess = 0
    fails = 0

    # Confusion Matrix
    confusion_matrix = {}
    for i in range(0, len(elastic_patterns)):
        confusion_matrix[str(i)] = [0] * len(elastic_patterns)

    for sample, target in zip(X_test, y_test):

        predicted_elastic_pattern = predict_elastic_pattern(sample, elastic_patterns)

        if predicted_elastic_pattern.meaning == target:
            sucess += 1
        else:
            fails += 1

        # update confusion matrix
        aux = confusion_matrix[str(target)]
        aux[int(predicted_elastic_pattern.meaning)] += 1

    accuracy = sucess * 100 / len(X_test)

    print("Nº of samples:", len(X_test), ",Nº of success: ", sucess, ", % of success: ", accuracy,"\n")

    print("--- Confusion matrix ----")
    for key, value in confusion_matrix.items():
        print(key, value)

    return accuracy

def sanitize_dataframe(df_1,df_2):

    df_1_clean = df_1.copy()
    df_2_clean = df_2.copy()

    for col in df_1_clean.columns:

        if isinstance(df_1[col].dtype, pd.CategoricalDtype):

            # Encode values
            unique_values = pd.concat([df_1[col], df_2[col]])

            ## Fit
            le = LabelEncoder()
            le.fit(unique_values)

            ## Apply encoding fro the two dataframes
            df_1_clean[col] = le.transform(df_1_clean[col])
            df_2_clean[col] = le.transform(df_2_clean[col])

    return df_1_clean, df_2_clean

def execute_experiment(deformation_method, experiment_name, group_function):

    start = time.time()
    print("Downloading the data...")
    X, y = fetch_openml('Adult-Census-Income', version=2, return_X_y=True)
    y = X['income']
    X = X.drop(columns = ['income'])

    print("Download complete!")
    print("Download time: ", time.time() - start, " seconds\n")

    print("=== RAW DATA ===")
    print("=== FEATURES ===")
    print(X)
    print(f"Features: {X.columns}")
    print(f"Nº of features: {len(X.columns)}")
    for col in X.columns:
        print(f"Col: {col} -> type {X[col].dtype}")
    print("=== TARGET ===")
    print(y)

    # Preproccess
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print(f"Nº of rows for training: {len(X_train)}")
    print(f"Nº of rows for test: {len(X_test)}")

    print("=== Converting Types ===")
    X_train_encoded, X_test_encoded = sanitize_dataframe(X_train, X_test)

    y_train_encoded = y_train.map({'=50K':1, '50K':0}) # 50K for over 50k yearly incom, =50K for 50k or less yearly
    y_train_encoded = y_train_encoded.astype(int)

    y_test_encoded = y_test.map({'=50K':1, '50K':0}) # 50K for over 50k yearly incom, =50K for 50k or less yearly
    y_test_encoded = y_test_encoded.astype(int)

    data = X_train_encoded
    data['income'] = y_train_encoded

    print("=== DATA ===")
    print(data)

    # Split data by target feature values
    data_over_threshold_income = data[data['income'] == 0]
    data_under_threshold_income = data[data['income'] == 1]

    print("=== DATA Over Threshold===")
    print(data_over_threshold_income)

    print("=== DATA Under Threshold===")
    print(data_under_threshold_income)

    #Group Function
    print(f"Group function: {group_function}")
    data_over_threshold_income_grouped = None
    data_under_threshold_income_grouped = None

    print(" === Data Over Threshold ===")
    print(data_over_threshold_income.describe())

    print(" === Data Under Threshold ===")
    print(data_under_threshold_income.describe())

    if group_function == "mean":
        data_over_threshold_income_grouped = data_over_threshold_income.mean().to_frame().T
        data_under_threshold_income_grouped = data_under_threshold_income.mean().to_frame().T

    if group_function == "min":
        data_over_threshold_income_grouped = data_over_threshold_income.min().to_frame().T
        data_under_threshold_income_grouped = data_under_threshold_income.min().to_frame().T

    if group_function == "max":
        data_over_threshold_income_grouped = data_over_threshold_income.max().to_frame().T
        data_under_threshold_income_grouped = data_under_threshold_income.max().to_frame().T

    data_grouped = pd.concat([data_over_threshold_income_grouped, data_under_threshold_income_grouped])

    print(" === Training Data ===")
    print(data_grouped)
    #for col in data_grouped.columns:
    #    print(f"Col: {col} -> type {data_grouped[col].dtype}")

    # Generate elastic pattern
    values = data_grouped.values

    elastic_patterns = []
    elastic_patterns.append(ElasticPattern(parameters = values[0], meaning = 1, deformation_method = deformation_method))
    elastic_patterns.append(ElasticPattern(parameters = values[1], meaning = 0, deformation_method = deformation_method))

    print(" === Comparing with samples ===")

    # Log to MLFlow
    mlflow_server_url = "http://host.docker.internal:5000"
    print(f"Connecting with MlFlow server: {mlflow_server_url}")
    print(f"Connecting to experiment: {experiment_name}")
    mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        print("Characterizing samples ...")
        start = time.time()
        accuracy = test_elastic_patterns(X_test_encoded.values, y_test_encoded.values, elastic_patterns)
        print("Execution time: ", time.time() - start, " seconds\n")
        print(f"Accuracy: {accuracy}")

        mlflow.log_param("deformation_method", deformation_method)
        mlflow.log_param("group_fucntion", group_function)
        mlflow.log_param("n_of_samples", len(X_test.values))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("Deformation Method", deformation_method)
        mlflow.set_tag("Experiment", "Elastic Pattern Incomes Threshold")
