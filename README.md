#  Worklfows for Automated Experimentation Platform

## Table of contents

1. [Project Structure](#project-structure)
2. [Experiments Description](#experiments-description)
3. [How to use](#how-to-use)


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

## Experiments Description

### Experiment 1

Creation of Elastic Pattern, and classification/characterization of samples with MNIST dataset, handwritten digits, more info about this datasets could be founde here: https://www.openml.org/d/554

Elastic Patterns using *Mask* concept: to obtain a
Elastic Pattern of a single digit, one can arrange the training samples on top of one another, thereby creating darker areas, which will be more important for recognition and therefore given more weight, and lighter areas, which will be less important for recognition and therefore given less weight. Figure below  illustrates this concept graphically.

![EP Creation Method](doc/images/Metodo.PNG)

Pool recomended to create/use: experiment_1_pool -> 2 slots

### Experiment 2

Creation of Elastic Pattern, and classification/characterization of samples with Wisconsin Breast Cancer dataset, more info about this datasets could be founde here: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

As an initial approach, we propose generating the Elastic Patterns using the arithmetic mean as the clustering function and using the following feature sets: All features,
Mean Values features, Worst Values features, and
Worst Values features. After that, another aggregation strategy, min and manx values are tested. One Elastic Pattern for conceptual category of the target feature *diagnosis*.

Pool recomended to create/use: experiment_2_pool -> 10 slots

### Experiment 3 

Creation of Elastic Pattern, and classification/characterization of samples with Adult Census Income, more info about this datasets could be founde here: https://www.openml.org/search?type=data&sort=runs&id=45565&status=active

As an initial approach, we propose generating the Elastic Patterns using different aggregation strategy, min and manx values are tested. One Elastic Pattern for conceptual category of the target feature *income*, one for incomes over 50k yearly and one for for incomes equals or under 50k yearly.

Pool recomended to create/use: experiment_3_pool -> 4 slots

## How To Use

This repository is and example of how to use workflows (as dags) inside the platform in this repo: https://github.com/RubenCardos/phd_platform, so the use could depends on the use case.

For experiment_1, dag: elastic_patterns_experiment_1.py its recomened to create a pool (in Airflow) called "experiment_1_pool" with 2 slots. More info about pools could be found here: https://www.astronomer.io/docs/learn/airflow-pools

