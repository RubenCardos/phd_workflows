import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

import mlflow

from phd_workflows.src.scripts.elastic_patterns_images_experiment import execute_experiment


EXPERIMENT_NAME = "elastic_patterns_experiment_1"

def check_experiment_exists():
    
    mlflow_server_url = "http://host.docker.internal:5000" #This could be a airflow vcariable or airflow connection
    mlflow.set_tracking_uri(mlflow_server_url)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Experiment {EXPERIMENT_NAME} created, id: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment {EXPERIMENT_NAME} already exists with id: {EXPERIMENT_NAME}")

DAG_ID = "elastic_patterns_experiment_1"

with DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2021, 1, 1),
    schedule=None,
):
    init = EmptyOperator(task_id = "init")
    end = EmptyOperator(task_id = "end")

    t_check_experiment_exists = PythonOperator(
                            task_id = f"check_experiment_{EXPERIMENT_NAME}_exists",
                            python_callable = check_experiment_exists,
                            pool = "experiment_1_pool"
    )

    for deformation_method in ["Hybrid", "Symmetric", "Asintotic", "Inverse"]:

        t_experiment = PythonOperator(
                            task_id=f"experiment_{deformation_method}_method", 
                            python_callable=execute_experiment, 
                            op_kwargs={"deformation_method": deformation_method, "experiment_name": EXPERIMENT_NAME},
                            pool = "experiment_1_pool"
                        )
        
        init >> t_check_experiment_exists >> t_experiment >> end



