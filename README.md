#  Worklfows for Automated Experimentation Platform

## Table of contents

1. [Project Structure](#project-structure)
1. [How to use](#how-to-use)

## Project Structure
The repository is organized as follows:


```bash
License  # License File, MIT license
README.md  # Readme file 
src # source code
.gitingore # git ignore file
├── scripts # folder to add Python script to execute inside the workflows (dags)
├── <expriment.py> # Python script to orquestate a experiment (could be define as a Python script in scripts folder) as a Airflow dag
└── ...
```

## How To Use

This repository is and example of how to use workflows (as dags) inside the platform in this repo: https://github.com/RubenCardos/phd_platform, so the use could depends on the use case.

For esperiment_1, dag: elastic_patterns_experiment_1.py its recomened to create a pool (in Airflow) called "experiment_1_pool" with 2 sloots. More info about pools could be found here: https://www.astronomer.io/docs/learn/airflow-pools

