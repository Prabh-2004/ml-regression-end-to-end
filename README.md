# House Price Prediction System

This project is an end-to-end machine learning application that predicts house prices based on various features like area, number of rooms, and other property details.

The goal of this project was to understand how to take a machine learning model from development to a usable system with an API.

---

## Features

- Data preprocessing using sklearn pipelines  
- Model training using Gradient Boosting Regressor  
- Experiment tracking using MLflow  
- Prediction pipeline for real-time inference  
- REST API built using Flask  
- Docker support for containerization  

---

## Project Structure

```
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── src/
│ ├── data_preprocessing.py
│ ├── train.py
│ └── predict.py
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```