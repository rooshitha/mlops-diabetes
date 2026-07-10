# 🩺 Diabetes Prediction using MLOps

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-success)

## 📌 Project Overview

This project is an end-to-end MLOps implementation for predicting diabetes progression using a Machine Learning model. It includes model training, REST API development with FastAPI, Docker containerization, and GitHub Actions for Continuous Integration (CI).

---

## 🚀 Tech Stack

- Python
- Scikit-Learn
- FastAPI
- MLflow
- Docker
- GitHub Actions
- Joblib
- NumPy
- Pandas

---

## ✨ Features

- Data preprocessing and model training
- Model serialization using Joblib
- REST API using FastAPI
- Interactive Swagger UI
- Dockerized application
- GitHub Actions CI pipeline

---

## 📂 Project Structure

```text
mlops-diabetes/
│── src/
│── tests/
│── app.py
│── train.py
│── requirements.txt
│── Dockerfile
│── README.md
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt

python train.py

uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

to access the Swagger API documentation.

---

## 🐳 Run with Docker

```bash
docker build -t diabetes-mlops .

docker run -p 8000:8000 diabetes-mlops
```

Then open:

```
http://localhost:8000/docs
```

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Home |
| GET | `/health` | Health Check |
| POST | `/predict` | Predict Diabetes Progression |

---

## 📜 License

This project is created for learning and educational purposes.