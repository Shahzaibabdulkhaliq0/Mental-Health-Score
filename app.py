import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="StudentMind | Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "Mental_Health_Model.pkl"

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 80% 0%,#17143b 0,#07111f 38%,#050b16 78%);color:#f4f7ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1427,#080e1b);border-right:1px solid #1d2a46}
.block-container{padding-top:1.2rem;max-width:1450px}
h1,h2,h3,p,label,span{color:#f4f7ff}
.hero{padding:1.5rem;border:1px solid #29345d;border-radius:22px;background:linear-gradient(110deg,rgba(92,48,190,.22),rgba(20,181,220,.08));margin-bottom:1.2rem}
.eyebrow{color:#50d9ef;font-size:.82rem;letter-spacing:2px;font-weight:700}
.hero h1{font-size:2.4rem;margin:.3rem 0}
.hero p,.muted{color:#aebed8}
.panel{border:1px solid #263451;border-radius:20px;padding:1.2rem;background:linear-gradient(145deg,rgba(16,27,48,.96),rgba(8,17,31,.96))}
.score{font-size:3.2rem;font-weight:800;background:linear-gradient(90deg,#c36bff,#24d9ef);-webkit-background-clip:text;color:transparent}
div.stButton>button{border:0;border-radius:12px;color:white;font-weight:700;min-height:3rem;background:linear-gradient(90deg,#923cff,#287dff);width:100%}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#111e34;border-color:#2a3c5d;border-radius:10px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 StudentMind")
    st.caption("Predict • Understand • Support")
    page = st.radio("NAVIGATION", ["🔮 Prediction", "ℹ️ About Project"])
    st.divider()
    st.markdown("### 💜 A gentle reminder")
    st.write("This model estimate is for educational use, not a medical diagnosis.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">MACHINE LEARNING · STUDENT WELLBEING</div>
  <h1>Student Mental <span style="color:#9a70ff">Health Predictor</span></h1>
  <p>Estimate a mental health score from student lifestyle and social-media-related factors.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload `Mental_Health_Model.pkl` to the same GitHub repository folder as `app.py`.")
    st.stop()

@st.cache_resource
def load_trained_model(model_path, modified_time):
    # Loads the already-trained notebook pipeline. No .fit() call happens in this app.
    return joblib.load(model_path)

try:
    model = load_trained_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    preprocessor = model.named_steps["preprocessor"]
    transformers = {name: transformer for name, transformer, cols in preprocessor.transformers_}
    cat_encoder = transformers["categorical"].named_steps["onehot"]
    cat_cols = ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use", "Grouped_country"]
    cat_options = {
        col: [str(v) for v in vals if not (isinstance(v, float) and np.isnan(v))]
        for col, vals in zip(cat_cols, cat_encoder.categories_)
    }
except Exception as e:
    st.error(f"Could not load the saved notebook model. Check that `Mental_Health_Model.pkl` was saved from `rf_pipeline`. Details: {e}")
    st.stop()

if page == "ℹ️ About Project":
    st.subheader("About this project")
    st.write("This app loads the saved `rf_pipeline` from the notebook.")
    st.markdown("""
    - **Model:** RandomForestRegressor (`random_state=42`)
    - **Preprocessing:** `Study_Hours` median imputation → log transform → scaling
    - **Stress level:** ordered encoding (Low, Medium, High, Very High)
    - **Other numeric fields:** median imputation and scaling
    - **Categorical fields:** most-frequent imputation and one-hot encoding
    - **Country:** mapped to `Grouped_country` using the notebook's top-country categories
    """)
    st.success("The deployed app only loads the trained model and calls predict. It does not retrain the model.")
else:
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("✦ Enter student details")
        st.caption("These inputs match the feature columns used by the saved notebook pipeline.")
        values = {}

        # Numeric feature names and ranges are aligned with the notebook's pipeline.
        numeric_defaults = {
            "Age": (15.0, 60.0, 22.0),
            "Study_Hours": (0.0, 24.0, 4.0),
            "Avg_Daily_Usage_Hours": (0.0, 24.0, 5.0),
            "Daily_Unlocks": (0.0, 500.0, 50.0),
            "Physical_Activity_Hours": (0.0, 24.0, 1.0),
            "Sleep_Hours_Per_Night": (0.0, 24.0, 7.0),
        }
        values["Age"] = st.number_input("Age", min_value=15.0, max_value=100.0, value=22.0, step=1.0)
        values["Gender"] = st.selectbox("Gender", cat_options["Gender"])
        values["Academic_Level"] = st.selectbox("Academic Level", cat_options["Academic_Level"])
        values["Study_Hours"] = st.number_input("Study Hours (per day)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        values["Avg_Daily_Usage_Hours"] = st.number_input("Average Daily Usage Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        values["Most_Used_Platform"] = st.selectbox("Most Used Platform", cat_options["Most_Used_Platform"])
        stress_opts = ["Low", "Medium", "High", "Very High"]
        values["Stress_Level"] = st.selectbox("Stress Level", stress_opts)
        values["Daily_Unlocks"] = st.number_input("Daily Unlocks", min_value=0.0, max_value=2000.0, value=50.0, step=1.0)
        values["Physical_Activity_Hours"] = st.number_input("Physical Activity Hours (per day)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        values["Sleep_Hours_Per_Night"] = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        values["Purpose_Of_Use"] = st.selectbox("Purpose of Use", cat_options["Purpose_Of_Use"])
        values["Grouped_country"] = st.selectbox("Country Group", cat_options["Grouped_country"])

        predict = st.button("✦ Predict Mental Health Score →")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("📈 Prediction result")
        st.caption("Prediction from your saved, trained Random Forest pipeline.")
        if predict:
            # IMPORTANT: column names/order match the X_train pipeline from the notebook.
            feature_order = [
                "Age", "Gender", "Academic_Level", "Study_Hours",
                "Avg_Daily_Usage_Hours", "Most_Used_Platform", "Stress_Level",
                "Daily_Unlocks", "Physical_Activity_Hours", "Sleep_Hours_Per_Night",
                "Purpose_Of_Use", "Grouped_country",
            ]
            input_df = pd.DataFrame([[values[col] for col in feature_order]], columns=feature_order)
            try:
                prediction = float(model.predict(input_df)[0])
                st.markdown(
                    f'<div class="panel"><div class="muted">PREDICTED MENTAL HEALTH SCORE</div>'
                    f'<div class="score">{prediction:.2f}</div>'
                    f'<div class="muted">Model estimate · not a diagnosis</div></div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Prediction failed. Please check that the uploaded model was saved from the notebook's `rf_pipeline`. Details: {e}")
        else:
            st.markdown(
                '<div class="panel"><div class="muted">READY WHEN YOU ARE</div>'
                '<div class="score">—</div><div class="muted">Complete the form and click Predict.</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("#### 💜 Wellbeing note")
        st.write("This score reflects patterns learned from your dataset. It cannot replace professional advice.")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("StudentMind · Educational ML project · Predictions are estimates, not medical advice.")
