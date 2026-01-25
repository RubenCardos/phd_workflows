from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
from math import e

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB


import mlflow

class ElasticPattern:

    def __init__(self, parameters, meaning, weights = [], deformation_method=4):
        self.parameters = parameters
        self.weights = weights
        self.meaning = meaning
        self.method = deformation_method

    def __str__(self):
        #return f"Elasctic Pattern meaning : {self.meaning}, deformation method: {self.method}, parametric representation ->{self.parameters}" 
        return f"Elastic Pattern meaning : {self.meaning}, deformation method: {self.method}"

    def compare(self, sample):

        deformation_vector = []

        for index in range(0, len (sample)):
           
            real_case = sample[index]
            parameter = self.parameters[index]

            engineering_strain = 0

            # Resolución de Cero Valores
            if parameter == 0:
                parameter = 1

            if real_case == 0:
                real_case = 1

            # Deformación axial Metodo 1 - Hibrido
            if self.method == "Hybrid":

                if real_case >= parameter:
                    engineering_strain = abs((real_case - parameter) / parameter)

                if parameter > real_case:
                    engineering_strain = abs((parameter - real_case) / real_case)

            # Hibrido inverso
            if self.method == "Inverse":

                if real_case >= parameter:
                    engineering_strain = abs((parameter - real_case) / real_case)

                if parameter > real_case:
                    engineering_strain = abs((real_case - parameter) / parameter)

            # Asintotico
            if self.method == "Asintotic":
                engineering_strain = abs((parameter - real_case) / real_case)

            # Simetrico
            if self.method == "Symmetric":
                engineering_strain = abs((real_case - parameter) / parameter)

            deformation_vector.append(engineering_strain) #Falta el peso del parametro aqui 

        deformation_energy = 0
        for j in deformation_vector:
            deformation_energy += j

        return deformation_energy


def predict_elastic_pattern(sample, elastic_patterns):
    res = -1
    weights = []

    for elastic_pattern in elastic_patterns:
        weights.append([elastic_pattern.compare(sample), elastic_pattern])

    min_weight_index = weights.index(min(weights))
    res = elastic_patterns[min_weight_index]

    # print(weights)
    # print(res,"\n")

    return res


# TODO normlizar y escalar las muestras y los patrones

def predict_masks(sample, masks):  # Masks es un diccionario {  1 => Mascara, etc...}
    res = -1

    weights = []
    for target, mask in masks.items():
        aux = 0
        for s_i, m_i in zip(sample, mask):
            if s_i != 255 and m_i != 255:  # Esto era asi puesto que las muestras era binarias, aqui no es asi
                aux += 255 - m_i
        weights.append([aux, target])

    min_weight = 9999999999999999999999999
    for weight in weights:
        if weight[0] < min_weight:
            min_weight = weight[0]
            res = weight[1]
    return res


def test_elastic_patterns(X_test, y_test, elastic_patterns):
    
    sucess = 0
    fails = 0

    # Confusion Matrix
    confusion_matrix = {}
    for i in range(0, 10):
        confusion_matrix[str(i)] = [0] * 10

    # Eliminar los errores almacenados de ejecuaciones anteriores
    # shutil.rmtree('resources/mnist/fails')
    # os.makedirs('resources/mnist/fails', 0o755)

    for sample, target in zip(X_test, y_test):
        
        predicted_elastic_pattern = predict_elastic_pattern(sample, elastic_patterns)

        # print("Target: ", target,"Predict: ", predict, " weights: ", weights)

        if predicted_elastic_pattern.meaning == target:
            sucess += 1
        else:
            fails += 1
            # save_fail(predict, target, sample, weights)

        # update confusion matrix
        aux = confusion_matrix[str(target)]
        aux[int(predicted_elastic_pattern.meaning)] += 1

    accuracy = sucess * 100 / len(X_test)

    print("Nº of samples:", len(X_test), ",Nº of success: ", sucess, ", % of success: ", accuracy,"\n")
    
    print("--- Confusion matrix ----")
    for key, value in confusion_matrix.items():
        print(key, value)

    return accuracy


def test_masks(X_test, y_test, masks):
    sucess = 0
    fails = 0

    for sample, target in zip(X_test, y_test):
        predict = predict_masks(sample, masks)
        if predict == target:
            sucess += 1
        else:
            fails += 1

    print("Nº of samples:", len(X_test), ",Nº of success: ", sucess, ", % of success: ", sucess * 100 / len(X_test))


def test_random_forests(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    success = 0
    for sample, value in zip(X_test, y_test):
        predict = clf.predict(sample.reshape(1, -1))
        if predict == value:
            success += 1

    print("Nº of samples:", len(X_test), ",Nº of success: ", success, ", % of success: ", success * 100 / len(X_test),
          "\n")


def test_knn(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    success = 0
    for sample, value in zip(X_test, y_test):
        predict = clf.predict(sample.reshape(1, -1))
        if predict == value:
            success += 1

    print("Nº of samples:", len(X_test), ",Nº of success: ", success, ", % of success: ", success * 100 / len(X_test),
          "\n")


def test_adaboost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    success = 0
    for sample, value in zip(X_test, y_test):
        predict = clf.predict(sample.reshape(1, -1))
        if predict == value:
            success += 1

    print("Nº of samples:", len(X_test), ",Nº of success: ", success, ", % of success: ", success * 100 / len(X_test),
          "\n")


def test_naive_bayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    success = 0
    for sample, value in zip(X_test, y_test):
        predict = clf.predict(sample.reshape(1, -1))
        if predict == value:
            success += 1

    print("Nº of samples:", len(X_test), ",Nº of success: ", success, ", % of success: ", success * 100 / len(X_test),
          "\n")


def execute_experiment(deformation_method, experiment_name):

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
        mlflow.set_tag("Experiment", "Elastic Pattern Images")