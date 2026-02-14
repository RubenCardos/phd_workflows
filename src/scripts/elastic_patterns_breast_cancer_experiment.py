from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
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
    for i in range(0, 10):
        confusion_matrix[str(i)] = [0] * 10

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


def execute_experiment(deformation_method, experiment_name):

    # Path inside platofrm/airflow for the file (as an example how to load data within the platform)
    data_path = "/opt/airflow/dags/phd_workflows/src/scripts/resources/breast_cancer/breast-cancer-wisconsin-data.csv"
    data = pd.read_csv(data_path, sep = ',')

    ## Drop id
    diagnosis = data['diagnosis']
    data = data.drop('id', axis = 1)
    data = data.drop('diagnosis', axis = 1)
    
    # Preproccess
    X_train, X_test, y_train, y_test = train_test_split(data, diagnosis, test_size=0.33)

    print(f"Nº of rows for training: {len(X_train)}")
    print(f"Nº of rows for test: {len(X_test)}")

    data = X_train
    data['diagnosis'] = y_train

    ## Map target feature "diagnosis"
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    data['diagnosis'] = data['diagnosis'].astype(int)

    # Split data by target feature values

    data_diagnosis_m = data[data['diagnosis'] == 1]
    data_diagnosis_b = data[data['diagnosis'] == 0]

    print(" === Data Diagnosis M ===")
    print(data_diagnosis_m.describe())

    print(" === Data Diagnosis B ===")
    print(data_diagnosis_b.describe())

    data_diagnosis_m_mean = data_diagnosis_m.mean().to_frame().T
    data_diagnosis_b_mean = data_diagnosis_b.mean().to_frame().T
    data_diagnosis_mean = pd.concat([data_diagnosis_m_mean, data_diagnosis_b_mean])

    print(" === Training Data ===")
    print(data_diagnosis_mean)

    # Generate elastic pattern
    values = data_diagnosis_mean.values

    elastic_patterns = []
    elastic_patterns.append(ElasticPattern(parameters = values[0], meaning = 1, deformation_method = deformation_method))
    elastic_patterns.append(ElasticPattern(parameters = values[1], meaning = 0, deformation_method = deformation_method))

    print(" === Comparing with samples ===")

    X_test ['diagnosis'] = y_test
    X_test['diagnosis'] = X_test['diagnosis'].map({'M':1, 'B':0})
    X_test['diagnosis'] = X_test['diagnosis'].astype(int)

    y_test= y_test.map({'M':1, 'B':0})

    # Log to MLFlow
    mlflow_server_url = "http://host.docker.internal:5000"
    print(f"Connecting with MlFlow server: {mlflow_server_url}")
    print(f"Connecting to experiment: {experiment_name}")
    mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        print("Characterizing samples ...")
        start = time.time()
        accuracy = test_elastic_patterns(X_test.values, y_test.values, elastic_patterns)
        print("Execution time: ", time.time() - start, " seconds\n")

        mlflow.log_param("deformation_method", deformation_method)
        mlflow.log_param("n_of_samples", len(X_test.values))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("Deformation Method", deformation_method)
        mlflow.set_tag("Experiment", "Elastic Pattern Breast Cancer")