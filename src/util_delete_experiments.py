
from __future__ import annotations
import textwrap
from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
import mlflow

def _delete_experiment(**context):
    
    conf = context["dag_run"].conf
    experiment_name = conf.get("experiment_name")
    
    mlflow_server_url = "http://host.docker.internal:5000" #This could be a airflow vcariable or airflow connection
    mlflow.set_tracking_uri(mlflow_server_url)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment {experiment_name} dont exists")
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment {experiment_name} already exists with id: {experiment_name}, deleting ...")
        mlflow.delete_experiment(experiment_id)
        print(f"Experiment {experiment_name} deleted")

with DAG(
    "utils_delete_experiment",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    # [END default_args]
    description="A utils DAG for delete MLflow experiments",
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    params={
        "experiment_name": Param("", type="string")
    },
    tags=["utils", "mlflow"],
) as dag:


    delete_experiment = PythonOperator(
        task_id="delete_experiment",
        python_callable = _delete_experiment
    )

    delete_experiment