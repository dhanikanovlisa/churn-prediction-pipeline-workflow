# Project Readme

This document provides step-by-step instructions for setting up and running the project. Follow the steps below to ensure everything runs smoothly.

## Prerequisites
- Docker and Docker Compose installed
- Basic familiarity with Airflow, Spark, and MLflow

---

## 1. Start Services
To start all necessary services, run the following command in the project root directory:

```bash
docker-compose up -d
```

> **Note:** Wait until all services are fully ready before proceeding to the next steps.

---

## 2. Access Services
Once the services are up and running, you can access the following interfaces:

- **Airflow**: [http://localhost:8080](http://localhost:8080)
- **MLflow**: [http://localhost:5050](http://localhost:5050)
- **Spark Master**: [http://localhost:8081](http://localhost:8081)

---

## 3. Configure Airflow Connection to Spark
To enable Airflow to interact with Spark, set up a connection in Airflow:

1. Open Airflow: [http://localhost:8080](http://localhost:8080)
2. Navigate to `Admin` > `Connections`.
![Navigate to `Admin` > `Connections`](images/(1).jpg)
3. Click **Add Connection** and configure it as follows:
   - **Connection ID**: `spark_default`
   - **Connection Type**: `Spark`
   - **Host**: `spark://spark-master:7077`
   - Other fields can remain default unless additional customization is needed.
![Configure Connection - 1](images/(2).jpg)
![Configure Connection - 2](images/(3).jpg)
4. Save the connection.

---

## 4. Trigger the DAG
To process data and train the model:

1. Open Airflow: [http://localhost:8080](http://localhost:8080)
2. Locate the relevant DAG and trigger it manually.
3. Monitor the DAG's execution to ensure it completes successfully.

---

## 5. Results
- **Processed Data**: The processed data will be saved in the `data/processed/` directory.
- **Model Deployment**: The trained model will be served and available in MLflow at [http://localhost:5050](http://localhost:5050).

---

## Troubleshooting
If you encounter issues:
- Verify all services are running by checking the Docker containers: `docker ps`
- Check logs for specific services using: `docker logs <container_name>`
- Ensure all connection configurations are correct in Airflow.

## Team Members
| Name | NIM |
| :------------- | :------------- |
| Tabitha Permalla | 13521111 |
| Althaaf Khasyi Atisomya | 13521130 |
| Dhanika Novlisariyanti | 13521132 |
