#  Worklfows for Automated Experimentation Platform

## Table of contents

1. [Project Structure](#project-structure)
1. [How to use](#how-to-use)

## Project Structure
The repository is organized as follows:


```bash
mlflow/
├── Dockerfile # Dockerfile to build amd Docker image for MLFlow server
├── requirements.txt   # Requirements file for MLFLow server
├── Dockerfile   # Dockerfile to create custom Airflow image, yo include git and all requeriments
├── compose.yml  # Docker Compose file to orchestrate the complete deployment of the platform
├── License  # License File, MIT license
├── README.md  # Readme file 
└── gitignore.yml # Git ignore file, to ignore some non-needed file in the reposotory
```

## How To Use

```bash
docker compose up -d 
```

ECPLICAR AQUI COMO SE USAN LOS DAS EN LA VERSION VERSIONADA Y SIN VERSIONAR