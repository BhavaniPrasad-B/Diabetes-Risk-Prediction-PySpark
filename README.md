# 🩺 Project : Diabetes Risk Insight Engine

A high-fidelity clinical analytics pipeline built with **PySpark** and **Streamlit**. This project predicts diabetes risk using the Pima Indians Diabetes Dataset with a distributed machine learning approach.

---

## 🏛️ Project Philosophy
This project is designed as an **End-to-End Data Engineering Pipeline**. It demonstrates the transition from raw, messy clinical data to a polished, "Aesthetic Royale" executive dashboard. By leveraging **Apache Spark**, the engine is built for horizontal scalability, capable of processing millions of patient records.

## 🚀 Key Technical Features
* **Scalable Ingestion:** Distributed data loading using PySpark SQL.
* **Biological Data Cleaning:** Custom imputation logic to handle logical zeros in Glucose, BMI, and Blood Pressure.
* **Feature Engineering:** Implementation of `VectorAssembler` and `StandardScaler` for magnitude parity.
* **Ensemble Learning:** A 100-tree **Random Forest Classifier** ensuring robust predictive power.
* **Aesthetic UI:** A bespoke Streamlit dashboard featuring Glassmorphism and Royal typography.

## 📊 Model Performance
The model is evaluated on a curated 20% test split, achieving consistent metrics:
* **F1-Score:** ~0.77
* **Precision:** ~0.79
* **Recall:** ~0.78

## 📂 Repository Structure
* `process.py`: The Spark engine (Cleaning, Feature Engineering, Training).
* `app.py`: The Royal Aesthetic Dashboard (Visualization & UI).
* `diabetes.csv`: The primary clinical dataset.
* `requirements.txt`: Environment dependencies.

## 🚦 Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/BhavaniPrasad-B/Diabetes-Risk-Prediction-PySpark.git](https://github.com/BhavaniPrasad-B/Diabetes-Risk-Prediction-PySpark.git)