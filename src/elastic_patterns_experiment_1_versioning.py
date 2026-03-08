from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess


BITBUCKET_REPO_URL = "https://github.com/RubenCardos/phd_experiments"
REPO_NAME = "phd_experiments"
SCRIPT_NAME = "src/elastic_patterns_images_experiment.py"

import mlflow

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

def _execute_experiment(deformation_method, experiment_name):
    subprocess.run(["python", f"/tmp/{REPO_NAME}/{SCRIPT_NAME}","-en",experiment_name,"-dm",deformation_method], check=True)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 9),
    "retries": 0,
}

DAG_ID = "elastic_patterns_experiment_1_versioning"

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    start_date=datetime(2021, 1, 1),
    schedule=None,
):
    init = EmptyOperator(task_id = "init")
    end = EmptyOperator(task_id = "end", trigger_rule = "one_success")

    clone_repo = BashOperator(
        task_id="clone_repo",
        bash_command=f"cd /tmp && git clone {BITBUCKET_REPO_URL}",
    )

    t_check_experiment_exists = PythonOperator(
                            task_id = f"check_experiment_{EXPERIMENT_NAME}_exists",
                            python_callable = check_experiment_exists,
                            pool = "experiment_1_pool"
    )

    cleanup_repo = BashOperator(
        task_id="cleanup_repo",
        bash_command=f"cd /tmp && rm -rf {REPO_NAME}",
    )

    cleanup_repo_becouse_fail = BashOperator(
        task_id="cleanup_repo_becouse_fail",
        bash_command=f"cd /tmp && rm -rf {REPO_NAME}",
        trigger_rule="all_failed"
    )

    for deformation_method in ["Hybrid", "Symmetric", "Asintotic", "Inverse"]:

        t_experiment = PythonOperator(
            task_id=f"experiment_{deformation_method}_method",
            python_callable=_execute_experiment,
            op_kwargs={"deformation_method": deformation_method, "experiment_name": EXPERIMENT_NAME},
            pool = "experiment_1_pool"
        )

        t_check_experiment_exists >> t_experiment >> [cleanup_repo, cleanup_repo_becouse_fail]

    init >> clone_repo >> t_check_experiment_exists >> t_experiment >> [cleanup_repo, cleanup_repo_becouse_fail] >> end
