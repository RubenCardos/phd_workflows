from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
from math import e

import mlflow

from ElasticPattern import ElasticPattern, MLFlow_EP

def predict_elastic_pattern(sample, elastic_patterns):
    res = -1
    weights = []

    for elastic_pattern in elastic_patterns:
        weight = elastic_pattern.compare(sample)
        print(weight)
        weights.append([weight, elastic_pattern])

    for w in weights:
        print(w)

    min_weight_index = weights.index(min(weights))
    res = elastic_patterns[min_weight_index]

    # print(weights)
    # print(res,"\n")

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

        # print("Target: ", target,"Predict: ", predict, " weights: ", weights)

        if predicted_elastic_pattern.meaning == target:
            sucess += 1
        else:
            fails += 1

        # Update confusion matrix
        aux = confusion_matrix[str(target)]
        aux[int(predicted_elastic_pattern.meaning)] += 1

    accuracy = sucess * 100 / len(X_test)

    print("Nº of samples:", len(X_test), ",Nº of success: ", sucess, ", % of success: ", accuracy,"\n")
    
    print("--- Confusion matrix ----")
    for key, value in confusion_matrix.items():
        print(key, value)

    return accuracy

def main():

    deformation_method = 4
    experiment_name = "Custom_Models"

    print("==== Arguments ====")
    print(f"Deformation method: {deformation_method}")
    print("==== ==== ====")

    # Dataset from https://www.openml.org/d/554
    start = time.time()
    print("Downloading the data...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print("Download complete!")
    print("Download time: ", time.time() - start, " seconds\n")

    # Split data
    start = time.time()
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, test_size=10000)
    print("Data splitted!")
    print("Splitting time: ", time.time() - start, " seconds\n")

    # Create Elastic Patters
    start = time.time()
    print("Creating Elasttic Patterns ...")

    inc = 255 / len(X_train.values)
    masks = {}
    print("The increment per pixel is: ", inc)

    print("Creating Aux data strctured ...")
    for i in range(0, 10):
        masks[str(i)] = [0.0] * len(X_train.values[0])
    print("Aux data structures created")
    
    print("Execution time: ", time.time() - start, " seconds\n")

    start = time.time()
    for data, target in zip(X_train.values, y_train.values):
        masks_aux = masks[target]
        for pixel in range(0, len(data)):
            if data[pixel] != 0:
                masks_aux[pixel] += inc

        masks[target] = masks_aux

    for target, mask in masks.items():
        mask_aux = mask
        for pixel in range(0, len(mask_aux)):
            mask_aux[pixel] = int(mask_aux[pixel])
        masks[target] = mask_aux

    print("Elasttic Patterns created!")
    print("Execution time: ", time.time() - start, " seconds\n")

    # Comparacion via enfoque multi-paramentrico
    elastic_patterns = []

    for meaning, parameters in masks.items():
        
        # Matriz de pesos
        w = [1/len(parameters)] * len(parameters) # 28 * 28 ravel matrix

        elastic_pattern = ElasticPattern(parameters = parameters, weights = w, meaning = meaning, deformation_method = deformation_method)
        elastic_patterns.append(elastic_pattern)
        print(str(elastic_pattern))
        print(parameters)
        print()


    

    # Log to MLFlow
    mlflow_server_url = "http://host.docker.internal:5000"
    print(f"Connecting with MlFlow server: {mlflow_server_url}")
    print(f"Connecting to experiment: {experiment_name}")
    mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        # Save Elastic Patterns as models

        """for ep in elastic_patterns:
            print(f"Saving EP for number: {ep.meaning}")
            
            model_path = f"MNIST_EP_{ep.meaning}"
            model_uri = f"runs:/{run_id}/{model_path}"

            #mlflow.pyfunc.save_model(path=model_path, python_model=MLFlow_EP(ep))
            mlflow.pyfunc.log_model(
                artifact_path=model_path,
                python_model=MLFlow_EP(ep),
                #signature=signature, #signature = infer_signature(X_train, lr.predict(X_train))
                #input_example=X_train,
            )
            result = mlflow.register_model(
                model_uri=model_uri, 
                name=model_path
            )
        """

        print("Characterizing samples ...")
        start = time.time()
        accuracy = test_elastic_patterns(X_test.values, y_test.values, elastic_patterns)
        print("Execution time: ", time.time() - start, " seconds\n")

        mlflow.log_param("deformation_method", deformation_method)
        mlflow.log_param("n_of_samples", len(X_test.values))
        mlflow.log_param("data_locator", 'mnist_784')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("Deformation Method", deformation_method)
        mlflow.set_tag("Experiment", "Elastic Pattern Images")

if __name__ == "__main__":
    main()