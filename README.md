# StudentMind — Mental Health Score Predictor

StudentMind is a machine-learning web app that estimates a student's `Mental_Health_Score` from lifestyle, study, and social-media-use information. The app is built with Streamlit and loads a previously trained model saved as `Mental_Health_Model.pkl`.

> **Disclaimer:** This project is for educational and portfolio purposes only. Its predictions are estimates from a dataset and are not a medical diagnosis or a substitute for professional advice.

## Live Demo

[Open StudentMind on Streamlit Community Cloud](https://mental-health-score-ltmvhz8vfl7xc8746dzeym.streamlit.app/)

## Dataset

The project uses the **Student Social Media and Mental Health Impact** dataset from Kaggle:

- Dataset: https://www.kaggle.com/datasets/shivasingh4945/student-social-media-and-mental-health-impact
- Target column: `Mental_Health_Score`

The saved pipeline expects these 12 input features:

- `Age`
- `Gender`
- `Academic_Level`
- `Most_Used_Platform`
- `Purpose_Of_Use`
- `Avg_Daily_Usage_Hours`
- `Daily_Unlocks`
- `Study_Hours`
- `Physical_Activity_Hours`
- `Sleep_Hours_Per_Night`
- `Stress_Level`
- `Grouped_country`

## Model and Evaluation

The saved notebook pipeline uses a Random Forest regressor and preprocessing steps from the training notebook. The Streamlit app loads the saved pipeline and calls `predict`; it does not train the model when the app starts.

Reported evaluation results:

| Metric | Result |
|---|---:|
| RMSE | 0.42 |
| R² | 0.90 |

These values are the project results provided by the author. Their interpretation depends on the evaluation split and procedure used during model development.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app and prediction interface |
| `Mental_Health_Model.pkl` | Saved trained model pipeline |
| `ML_Project.ipynb` | Notebook for data analysis and model development |
| `Student Social Media And Mental Health Impact.csv` | Dataset file used by the project |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python runtime configuration for deployment |

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shahzaibabdulkhaliq0/Mental-Health-Score.git
cd Mental-Health-Score
```

### 2. Create and activate a Conda environment

```bash
conda create -n machine_learning python=3.12 -y
conda activate machine_learning
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the app

Make sure `Mental_Health_Model.pkl` is in the same directory as `app.py`, then run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`).

## How to Use

1. Open the Prediction page.
2. Enter the requested student lifestyle and social-media-use details.
3. Select the available categorical values.
4. Click **Predict Mental Health Score**.
5. Review the estimated score and the educational-use reminder.

## Reproducibility Notes

- The app relies on the serialized pipeline in `Mental_Health_Model.pkl`.
- Keep the model file and `app.py` in the same folder.
- Use the dependency versions recorded in `requirements.txt` when running or deploying the app.
- Loading a pickle/joblib model with incompatible library versions can cause errors. If the model cannot load, compare the scikit-learn and joblib versions used to save it with the environment used to run the app.

## Limitations

- The model learns patterns from its training dataset; estimates may not generalize to every student or population.
- A high R² score does not establish clinical validity.
- The app should not be used to diagnose, screen, or make high-stakes decisions about an individual's mental health.

## Acknowledgements

- Dataset: [Student Social Media and Mental Health Impact — Kaggle](https://www.kaggle.com/datasets/shivasingh4945/student-social-media-and-mental-health-impact)
- App framework: [Streamlit](https://streamlit.io/)
