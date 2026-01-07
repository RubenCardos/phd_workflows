import time
from math import e
import mlflow

# Your imports here

class ElasticPattern:

    def __init__(self, parameters, meaning, weights = [], deformation_method=4):
        self.parameters = parameters
        self.weights = weights
        self.meaning = meaning
        self.method = deformation_method

    def __str__(self):
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


def execute_experiment(deformation_method, experiment_name):

    print("==== Arguments ====")
    print(f"Deformation method: {deformation_method}")
    print("==== ==== ====")

    #################
    # You code here #
    #################
    
    # MLFlow connection config
    mlflow_server_url = "http://host.docker.internal:5000"
    print(f"Connecting with MlFlow server: {mlflow_server_url}")
    print(f"Connecting to experiment: {experiment_name}")
    mlflow.set_tracking_uri(mlflow_server_url)
    # mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        print("Saving params and matrics in MLFlow")

        ########################################################
        # The parameters and metrics you want to register here #
        ########################################################

        # mlflow.log_param("deformation_method", deformation_method)
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.set_tag("Deformation Method", deformation_method)