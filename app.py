import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="StudentMind | Mental Health Predictor", page_icon="🧠", layout="wide")

MODEL_PATH = Path(__file__).parent / "Mental_Health_Model.pkl"

# Maximum possible score of the target column. Change to 100 if your score is out of 100.
MAX_SCORE = 10.0

# (minimum percent, label, colour, face, message)
BANDS = [
    (80, "Excellent Mental Health", "#22d3a6", "😄", "You are managing your well-being very well. Keep it up!"),
    (60, "Good Mental Health", "#24d9ef", "🙂", "You seem to be managing well. Keep maintaining a healthy lifestyle."),
    (40, "Moderate Mental Health", "#f5c542", "😐", "There is room to improve. Look at your sleep, screen time and breaks."),
    (20, "Low Mental Health", "#f59e42", "😟", "Take extra care of yourself and talk to someone you trust."),
    (0, "Poor Mental Health", "#f0506e", "😢", "Please consider reaching out to a counsellor or doctor for support."),
]

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 80% 0%,#17143b 0,#07111f 38%,#050b16 78%);color:#f4f7ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1427,#080e1b);border-right:1px solid #1d2a46}
.block-container{padding-top:1rem;max-width:1450px}
h1,h2,h3,h4,p,label,span,li{color:#f4f7ff}
.topbar{padding:.8rem 1.2rem;border-bottom:1px solid #1d2a46;margin-bottom:1.2rem}
.logo{font-size:1.6rem;font-weight:800}
.logo span{color:#a46bff}
.logo small{display:block;font-size:.72rem;font-weight:400;color:#aebed8}
.hero h1{font-size:2.6rem;margin:0}
.hero h1 span{color:#6f8bff}
.muted{color:#aebed8}
.feat{display:flex;gap:2rem;margin:1rem 0 1.4rem 0;flex-wrap:wrap}
.feat div{font-size:.9rem}
.feat b{display:block}
.panel{border:1px solid #263451;border-radius:20px;padding:1.2rem;background:linear-gradient(145deg,rgba(16,27,48,.96),rgba(8,17,31,.96));margin-bottom:1rem}
.ring{width:190px;height:190px;border-radius:50%;margin:auto;display:flex;align-items:center;justify-content:center}
.ring-inner{width:150px;height:150px;border-radius:50%;background:#0b1427;display:flex;flex-direction:column;align-items:center;justify-content:center}
.big{font-size:3.2rem;font-weight:800;line-height:1}
.result{border-radius:14px;padding:1rem;margin-top:.8rem}
.band{display:flex;gap:.8rem;font-size:.85rem;margin:.3rem 0;padding:.25rem .4rem;border-radius:8px;border:1px solid transparent}
.band.current{border-color:#6f8bff;background:rgba(111,139,255,.12)}
.tag{border-radius:6px;padding:0 .5rem;min-width:70px;text-align:center;color:#06101f;font-weight:700}
.sugg{border:1px solid #3a2a6b;background:rgba(90,50,170,.15);border-radius:14px;padding:1rem;margin-top:1rem}
.sugg ul{margin:.4rem 0 0 1.1rem;padding:0}
.flow{display:flex;flex-direction:column;align-items:center}
.step{width:100%;border:1px solid #263451;border-radius:14px;padding:.8rem 1.1rem;background:rgba(16,27,48,.9)}
.step b{color:#50d9ef}
.arrow{font-size:1.4rem;color:#6f8bff;line-height:1.4}
div.stButton>button{border:0;border-radius:12px;color:white;font-weight:700;min-height:3rem;background:linear-gradient(90deg,#923cff,#287dff);width:100%}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#111e34;border-color:#2a3c5d;border-radius:10px}

/* ---------- Animations ---------- */
@property --p{syntax:'<number>';inherits:false;initial-value:0}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideDown{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fillRing{from{--p:0}to{--p:var(--target)}}
@keyframes popIn{0%{opacity:0;transform:scale(.4)}70%{transform:scale(1.15)}100%{opacity:1;transform:scale(1)}}
@keyframes glow{0%,100%{box-shadow:0 0 12px rgba(36,217,239,.25)}50%{box-shadow:0 0 34px rgba(36,217,239,.65)}}
@keyframes gradMove{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes bandIn{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}

.topbar{animation:slideDown .7s ease both}
.hero h1{animation:fadeUp .8s ease both}
.hero p{animation:fadeUp .8s .15s ease both}
.feat div{animation:fadeUp .8s ease both;transition:transform .3s}
.feat div:nth-child(1){animation-delay:.25s}
.feat div:nth-child(2){animation-delay:.4s}
.feat div:nth-child(3){animation-delay:.55s}
.feat div:hover{transform:translateY(-6px)}
.panel{animation:fadeUp .8s .3s ease both;transition:transform .3s,border-color .3s}
.panel:hover{transform:translateY(-4px);border-color:#5b4bd6}
.step{animation:fadeUp .7s ease both;transition:transform .3s,border-color .3s}
.step:hover{transform:translateX(6px);border-color:#5b4bd6}
.ring-inner .big{animation:popIn .9s .4s ease both}
.result{animation:fadeUp .8s 1s ease both}
.sugg{animation:fadeUp .8s .6s ease both}
.band{animation:bandIn .6s ease both}
.band:nth-of-type(1){animation-delay:.5s}
.band:nth-of-type(2){animation-delay:.65s}
.band:nth-of-type(3){animation-delay:.8s}
.band:nth-of-type(4){animation-delay:.95s}
.band:nth-of-type(5){animation-delay:1.1s}
div.stButton>button{background-size:200% 200%;animation:gradMove 3s ease infinite;transition:transform .2s,box-shadow .2s}
div.stButton>button:hover{transform:translateY(-3px) scale(1.02);box-shadow:0 8px 24px rgba(146,60,255,.5)}
div.stButton>button:active{transform:scale(.97)}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧠 StudentMind")
    st.caption("Predict • Understand • Support")
    page = st.radio("NAVIGATION", ["🔮 Prediction", "ℹ️ About Project", "💡 Mental Health Tips"])
    st.divider()
    st.caption("This tool gives a model estimate for educational use. It is not a medical diagnosis.")

# ---------- Top bar ----------
st.markdown("""
<div class="topbar">
  <div class="logo">Student<span>Mind</span><small>Predict • Understand • Support</small></div>
</div>
""", unsafe_allow_html=True)

# ---------- About page (does not need the model) ----------
if page == "ℹ️ About Project":
    st.markdown("""
    <div class="hero">
      <h1>About <span>the Project</span></h1>
      <p class="muted">How the data becomes a prediction, from raw dataset to this app.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔄 Project Flow")
    steps = [
        ("1. Dataset", "A student survey dataset with lifestyle and social-media usage columns. The target is <b>Mental_Health_Score</b>."),
        ("2. Data Cleaning & Exploration", "Check missing values, duplicates and data types, then explore how sleep, screen time, stress and activity relate to the score."),
        ("3. Feature Engineering", "Country is reduced to the top countries and the rest are grouped, creating <b>Grouped_country</b>. This keeps the one-hot encoding small."),
        ("4. Train / Test Split", "The data is split so the model is evaluated on rows it has never seen."),
        ("5. Preprocessing Pipeline", "A <b>ColumnTransformer</b> handles every column type: numeric, ordinal (Stress Level) and categorical. See the table below."),
        ("6. Model Training", "A <b>RandomForestRegressor</b> (<code>random_state=42</code>) is attached to the preprocessor in one scikit-learn <b>Pipeline</b> called <code>rf_pipeline</code>."),
        ("7. Evaluation", "The pipeline is scored on the test set with regression metrics such as MAE, RMSE and R²."),
        ("8. Save the Model", "The full fitted pipeline is saved with <code>joblib.dump</code> as <b>Mental_Health_Model.pkl</b>."),
        ("9. Streamlit App", "This app loads the saved pipeline (it never retrains), turns your inputs into a one-row DataFrame and calls <code>predict</code>."),
        ("10. Result", "The score is shown on a ring, mapped to a health band and paired with simple lifestyle suggestions."),
    ]
    html = '<div class="flow">'
    index = 0
    for title, desc in steps:
        html += f'<div class="step" style="animation-delay:{index*0.08}s"><b>{title}</b><br><span class="muted">{desc}</span></div>'
        if index < len(steps) - 1:
            html += '<div class="arrow">↓</div>'
        index += 1
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("🧪 Preprocessing Pipeline")
    pipeline_table = pd.DataFrame({
        "Feature": ["Age", "Study_Hours", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Physical_Activity_Hours",
                    "Sleep_Hours_Per_Night", "Stress_Level", "Gender", "Academic_Level", "Most_Used_Platform",
                    "Purpose_Of_Use", "Grouped_country"],
        "Type": ["Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Ordinal",
                 "Categorical", "Categorical", "Categorical", "Categorical", "Categorical"],
        "Processing": ["Median imputation → Scaling",
                       "Median imputation → Log transform → Scaling",
                       "Median imputation → Scaling", "Median imputation → Scaling",
                       "Median imputation → Scaling", "Median imputation → Scaling",
                       "Ordered encoding (Low < Medium < High < Very High)",
                       "Most-frequent imputation → One-hot", "Most-frequent imputation → One-hot",
                       "Most-frequent imputation → One-hot", "Most-frequent imputation → One-hot",
                       "Most-frequent imputation → One-hot"],
    })
    st.dataframe(pipeline_table, use_container_width=True, hide_index=True)

    st.subheader("📁 Deployment Files")
    st.code("app.py                    # this Streamlit app\nMental_Health_Model.pkl   # saved rf_pipeline\nrequirements.txt          # streamlit, pandas, numpy, scikit-learn, joblib", language="text")
    st.info("Use the same scikit-learn version in requirements.txt as the one used to train the model, otherwise loading the .pkl can fail.")

    st.subheader("⚠️ Limitations")
    st.write("The model learns patterns from one survey dataset, so it only estimates a score. It cannot diagnose anything and should never replace professional advice.")
    st.stop()

# ---------- Tips page ----------
if page == "💡 Mental Health Tips":
    st.markdown("""
    <div class="hero">
      <h1>Mental Health <span>Tips</span></h1>
      <p class="muted">Small daily habits that support student well-being.</p>
    </div>
    """, unsafe_allow_html=True)
    tips_page = [
        ("😴 Sleep", "Aim for 7 to 9 hours every night and keep a regular sleep time."),
        ("📱 Screen time", "Take a short break every hour and keep your phone away from the bed."),
        ("🏃 Movement", "Even a 20 to 30 minute walk each day improves mood and focus."),
        ("📚 Study routine", "Study in focused blocks with breaks instead of long last-minute sessions."),
        ("🤝 Connection", "Talk to friends or family regularly. Do not carry stress alone."),
        ("🆘 Get help", "If you feel low for many days, speak to a counsellor, doctor or someone you trust."),
    ]
    for title, text in tips_page:
        st.markdown(f'<div class="panel"><b>{title}</b><br><span class="muted">{text}</span></div>', unsafe_allow_html=True)
    st.stop()

# ---------- Prediction page ----------
st.markdown("""
<div class="hero">
  <h1>Student Mental <span>Health Predictor</span></h1>
  <p class="muted">Understand your mental well-being with the power of Machine Learning.</p>
  <div class="feat">
    <div><b>✨ Quick Prediction</b><span class="muted">Get instant results</span></div>
    <div><b>📊 ML Powered</b><span class="muted">Random Forest model</span></div>
    <div><b>🛡️ Private</b><span class="muted">Nothing is stored</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload `Mental_Health_Model.pkl` to the same folder as `app.py`.")
    st.stop()

@st.cache_resource
def load_trained_model(model_path, modified_time):
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
    st.error(f"Could not load the saved model. Details: {e}")
    st.stop()

left, right = st.columns([1.1, .9], gap="large")
values = {}

with left:
    st.markdown("### ✦ Enter Your Details")
    st.caption("Provide the following information to predict your mental health score.")

    c1, c2 = st.columns(2)
    values["Age"] = c1.slider("Age", 15, 60, 22)
    values["Gender"] = c2.selectbox("Gender", cat_options["Gender"])

    c1, c2 = st.columns(2)
    values["Academic_Level"] = c1.selectbox("Academic Level", cat_options["Academic_Level"])
    values["Stress_Level"] = c2.selectbox("Stress Level", ["Low", "Medium", "High", "Very High"])

    c1, c2 = st.columns(2)
    values["Sleep_Hours_Per_Night"] = c1.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.5)
    values["Study_Hours"] = c2.slider("Study Hours (per day)", 0.0, 12.0, 4.0, 0.5)

    c1, c2 = st.columns(2)
    values["Avg_Daily_Usage_Hours"] = c1.slider("Social Media / Screen Time (hours)", 0.0, 12.0, 5.0, 0.5)
    values["Physical_Activity_Hours"] = c2.slider("Physical Activity (hours)", 0.0, 6.0, 1.0, 0.5)

    c1, c2 = st.columns(2)
    values["Daily_Unlocks"] = c1.number_input("Daily Phone Unlocks", 0, 500, 50, 1)
    values["Most_Used_Platform"] = c2.selectbox("Most Used Platform", cat_options["Most_Used_Platform"])

    c1, c2 = st.columns(2)
    values["Purpose_Of_Use"] = c1.selectbox("Purpose of Use", cat_options["Purpose_Of_Use"])
    values["Grouped_country"] = c2.selectbox("Country Group", cat_options["Grouped_country"])

    predict = st.button("✦ Predict Mental Health Score →")

with right:
    st.markdown("### 📈 Prediction Result")
    st.caption("Here is your predicted mental health score.")

    # Sleep + study + exercise cannot add up to more than 24 hours in a day.
    busy_hours = values["Sleep_Hours_Per_Night"] + values["Study_Hours"] + values["Physical_Activity_Hours"]
    current_band = None

    if predict and busy_hours > 24:
        st.error(f"Sleep, study and physical activity add up to {busy_hours:.1f} hours, which is more than a day. Please correct your inputs.")
    elif predict:
        feature_order = [
            "Age", "Gender", "Academic_Level", "Study_Hours",
            "Avg_Daily_Usage_Hours", "Most_Used_Platform", "Stress_Level",
            "Daily_Unlocks", "Physical_Activity_Hours", "Sleep_Hours_Per_Night",
            "Purpose_Of_Use", "Grouped_country",
        ]
        input_df = pd.DataFrame([[values[col] for col in feature_order]], columns=feature_order)
        try:
            prediction = float(model.predict(input_df)[0])
            percent = max(0.0, min(100.0, prediction / MAX_SCORE * 100))

            for band in BANDS:
                if percent >= band[0]:
                    current_band = band
                    break
            band_min, label, colour, face, message = current_band

            st.markdown(f"""
            <div class="panel">
              <div class="ring" style="--target:{percent*3.6};background:conic-gradient({colour} calc(var(--p)*1deg),#1d2a46 0deg);animation:fillRing 1.6s ease-out forwards;box-shadow:0 0 24px {colour}55">
                <div class="ring-inner">
                  <div class="big">{prediction:.1f}</div>
                  <div class="muted">/ {MAX_SCORE:.0f}</div>
                </div>
              </div>
              <h4 style="text-align:center;margin-top:.8rem">Mental Health Score</h4>
              <div class="result" style="border:1px solid {colour};background:{colour}22"><b>{face} {label}</b><br><span class="muted">{message}</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Personalised suggestions based on what the user entered
            tips = []
            if values["Sleep_Hours_Per_Night"] < 7:
                tips.append("Try to sleep 7 to 9 hours each night.")
            if values["Avg_Daily_Usage_Hours"] > 6:
                tips.append("Your screen time is high. Take regular breaks and set app limits.")
            if values["Physical_Activity_Hours"] < 0.5:
                tips.append("Add a short daily walk or light exercise.")
            if values["Stress_Level"] in ["High", "Very High"]:
                tips.append("Your stress is high. Talk to someone you trust or try relaxation breaks.")
            if values["Daily_Unlocks"] > 150:
                tips.append("You unlock your phone very often. Try turning off non-essential notifications.")
            if values["Study_Hours"] > 9:
                tips.append("Long study hours need rest. Study in blocks with breaks.")
            if len(tips) == 0:
                tips.append("Your habits look balanced. Keep your sleep, study and activity routine going.")
            tips_html = "".join([f"<li>{t}</li>" for t in tips])
            st.markdown(f'<div class="sugg"><b>💗 Suggestions for you</b><ul class="muted">{tips_html}</ul></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed. Details: {e}")
    else:
        st.markdown("""
        <div class="panel" style="text-align:center">
          <div class="muted">READY WHEN YOU ARE</div>
          <div class="big">—</div>
          <div class="muted">Fill in the form and click Predict.</div>
        </div>
        """, unsafe_allow_html=True)

    bands_html = ""
    ranges = ["80 - 100%", "60 - 79%", "40 - 59%", "20 - 39%", "0 - 19%"]
    index = 0
    for band in BANDS:
        css = "band current" if (current_band is not None and band[0] == current_band[0]) else "band"
        bands_html += f'<div class="{css}"><span class="tag" style="background:{band[2]}">{ranges[index]}</span>{band[1]}</div>'
        index += 1
    st.markdown(f'<div class="panel"><b>💡 Score Interpretation</b>{bands_html}</div>', unsafe_allow_html=True)

st.divider()
st.caption("StudentMind · Educational ML project · Predictions are estimates, not medical advice.")
