import mlflow.pyfunc

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

            deformation_vector.append(engineering_strain)

        deformation_energy = 0
        for j in deformation_vector:
            deformation_energy += j

        return deformation_energy