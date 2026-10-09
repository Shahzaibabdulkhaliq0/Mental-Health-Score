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

# ---------- Compact layout: everything fits one screen at 100% browser zoom ----------
# html font-size shrinks every rem-based Streamlit size (fonts, paddings, widgets) without using CSS zoom.
st.markdown("""
<style>
html{font-size:13px}
.stApp{background:radial-gradient(circle at 80% 0%,#17143b 0,#07111f 38%,#050b16 78%);color:#f4f7ff}
[data-testid="stHeader"]{display:none}
#MainMenu,footer{visibility:hidden}
.block-container,[data-testid="stMainBlockContainer"]{padding:.8rem 1.5rem 0 1.5rem!important;max-width:100%!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1427,#080e1b);border-right:1px solid #1d2a46}
[data-testid="stSidebar"][aria-expanded="true"]{width:190px!important;min-width:190px!important;max-width:190px!important}
[data-testid="stSidebarContent"]{padding-top:.5rem}
[data-testid="stVerticalBlock"]{gap:.45rem}
[data-testid="stHorizontalBlock"]{gap:.8rem}
[data-testid="stWidgetLabel"] p{font-size:.8rem;margin-bottom:0;color:#cfd9ee}
h1,h2,h3,h4,p,label,span,li{color:#f4f7ff}
.muted{color:#aebed8}
.hero{margin-bottom:.4rem}
.hero h1{font-size:1.7rem;margin:0;padding:0;line-height:1.2}
.hero h1 span{color:#6f8bff}
.hero p{font-size:.85rem;margin:.1rem 0 0 0}
.sec{font-size:1rem;font-weight:700;margin:0 0 .1rem 0}
.panel{border:1px solid #263451;border-radius:14px;padding:.7rem .9rem;background:linear-gradient(145deg,rgba(16,27,48,.96),rgba(8,17,31,.96))}
.row{display:flex;gap:.9rem;align-items:center}
.rtext{flex:1}
.ring{width:118px;height:118px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ring-inner{width:92px;height:92px;border-radius:50%;background:#0b1427;display:flex;flex-direction:column;align-items:center;justify-content:center}
.big{font-size:2.1rem;font-weight:800;line-height:1}
.result{border-radius:10px;padding:.45rem .6rem;margin-top:.4rem;font-size:.8rem}
.band{display:flex;gap:.6rem;font-size:.78rem;margin:.18rem 0;padding:.12rem .3rem;border-radius:8px;border:1px solid transparent;align-items:center}
.band.current{border-color:#6f8bff;background:rgba(111,139,255,.12)}
.tag{border-radius:6px;padding:0 .4rem;min-width:66px;text-align:center;color:#06101f;font-weight:700}
.sugg{border:1px solid #3a2a6b;background:rgba(90,50,170,.15);border-radius:12px;padding:.5rem .8rem;font-size:.8rem}
.sugg ul{margin:.2rem 0 0 1rem;padding:0}
.grid5{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem}
.step{border:1px solid #263451;border-radius:12px;padding:.55rem .7rem;background:rgba(16,27,48,.9);font-size:.76rem}
.step b{color:#50d9ef;display:block;margin-bottom:.15rem;font-size:.82rem}
.tbl{width:100%;border-collapse:collapse;font-size:.78rem}
.tbl th{text-align:left;color:#50d9ef;border-bottom:1px solid #263451;padding:.25rem .5rem}
.tbl td{border-bottom:1px solid #1a2744;padding:.2rem .5rem;color:#d7e0f2}
div.stButton>button{border:0;border-radius:10px;color:white;font-weight:700;min-height:2.4rem;background:linear-gradient(90deg,#923cff,#287dff);width:100%}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#111e34;border-color:#2a3c5d;border-radius:8px;min-height:2.3rem}
.stTabs [data-baseweb="tab-list"]{gap:.3rem}
.stTabs [data-baseweb="tab"]{padding:.3rem .8rem}

/* ---------- Animations that always work (result area, hover, button) ---------- */
@property --p{syntax:'<number>';inherits:false;initial-value:0}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes fillRing{from{--p:0}to{--p:var(--target)}}
@keyframes popIn{0%{opacity:0;transform:scale(.4)}70%{transform:scale(1.15)}100%{opacity:1;transform:scale(1)}}
@keyframes gradMove{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes bandIn{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
.panel,.step{transition:transform .3s,border-color .3s}
.panel:hover,.step:hover{transform:translateY(-3px);border-color:#5b4bd6}
.ring-inner .big{animation:popIn .9s .4s ease both}
.result{animation:fadeUp .8s 1s ease both}
.sugg{animation:fadeUp .8s .6s ease both}
div.stButton>button{background-size:200% 200%;animation:gradMove 3s ease infinite;transition:transform .2s,box-shadow .2s}
div.stButton>button:hover{transform:translateY(-2px) scale(1.02);box-shadow:0 8px 24px rgba(146,60,255,.5)}
div.stButton>button:active{transform:scale(.97)}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🧠 StudentMind")
    st.caption("Predict • Understand • Support")
    page = st.radio("NAVIGATION", ["🔮 Prediction", "ℹ️ About Project", "💡 Mental Health Tips"])
    st.divider()
    st.caption("Model estimate for educational use. Not a medical diagnosis.")

# Page-load animations play only when a page is first opened, not on every widget rerun.
# (On Streamlit Cloud every slider change reruns the app, which made animations replay and look broken.)
animate_load = st.session_state.get("last_page") != page
st.session_state["last_page"] = page

if animate_load:
    st.markdown("""
    <style>
    .hero h1{animation:fadeUp .8s ease both}
    .hero p{animation:fadeUp .8s .15s ease both}
    .panel{animation:fadeUp .8s .3s ease both}
    .step{animation:fadeUp .7s ease both}
    .step:nth-child(2n){animation-delay:.1s}
    .step:nth-child(3n){animation-delay:.2s}
    .band{animation:bandIn .6s ease both}
    .band:nth-of-type(1){animation-delay:.5s}
    .band:nth-of-type(2){animation-delay:.65s}
    .band:nth-of-type(3){animation-delay:.8s}
    .band:nth-of-type(4){animation-delay:.95s}
    .band:nth-of-type(5){animation-delay:1.1s}
    </style>
    """, unsafe_allow_html=True)

# ---------- About page (does not need the model) ----------
if page == "ℹ️ About Project":
    st.markdown('<div class="hero"><h1>About <span>the Project</span></h1><p class="muted">How the data becomes a prediction, from raw dataset to this app.</p></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔄 Project Flow", "🧪 Preprocessing Pipeline", "📁 Files & Limitations"])

    with tab1:
        steps = [
            ("1. Dataset", "A student survey dataset with lifestyle and social-media columns. The target is <b>Mental_Health_Score</b>."),
            ("2. Cleaning & Exploration", "Check missing values, duplicates and data types, then explore how sleep, screen time, stress and activity relate to the score."),
            ("3. Feature Engineering", "Country is reduced to the top countries and the rest are grouped into <b>Grouped_country</b>, keeping one-hot encoding small."),
            ("4. Train / Test Split", "Data is split so the model is evaluated on rows it has never seen."),
            ("5. Preprocessing Pipeline", "A <b>ColumnTransformer</b> handles numeric, ordinal (Stress Level) and categorical columns."),
            ("6. Model Training", "A <b>RandomForestRegressor</b> (random_state=42) is joined with the preprocessor in one Pipeline: <b>rf_pipeline</b>."),
            ("7. Evaluation", "The pipeline is scored on the test set with regression metrics such as MAE, RMSE and R²."),
            ("8. Save the Model", "The fitted pipeline is saved with joblib as <b>Mental_Health_Model.pkl</b>."),
            ("9. Streamlit App", "The app loads the saved pipeline (no retraining), builds a one-row DataFrame from your inputs and calls predict."),
            ("10. Result", "The score is shown on a ring, mapped to a health band and paired with simple lifestyle suggestions."),
        ]
        html = '<div class="grid5">'
        for title, desc in steps:
            html += f'<div class="step"><b>{title}</b><span class="muted">{desc}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    with tab2:
        rows = [
            ("Age", "Numeric", "Median imputation → Scaling"),
            ("Study_Hours", "Numeric", "Median imputation → Log transform → Scaling"),
            ("Avg_Daily_Usage_Hours", "Numeric", "Median imputation → Scaling"),
            ("Daily_Unlocks", "Numeric", "Median imputation → Scaling"),
            ("Physical_Activity_Hours", "Numeric", "Median imputation → Scaling"),
            ("Sleep_Hours_Per_Night", "Numeric", "Median imputation → Scaling"),
            ("Stress_Level", "Ordinal", "Ordered encoding (Low < Medium < High < Very High)"),
            ("Gender, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Grouped_country", "Categorical", "Most-frequent imputation → One-hot encoding"),
        ]
        table = '<table class="tbl"><tr><th>Feature</th><th>Type</th><th>Processing</th></tr>'
        for feature, kind, how in rows:
            table += f'<tr><td>{feature}</td><td>{kind}</td><td>{how}</td></tr>'
        table += '</table>'
        st.markdown(table, unsafe_allow_html=True)

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📁 Deployment files**")
            st.code("app.py\nMental_Health_Model.pkl\nrequirements.txt", language="text")
        with c2:
            st.markdown("**📌 Version note**")
            st.write("Use the same scikit-learn version in requirements.txt as the one used for training, otherwise the .pkl may fail to load.")
        with c3:
            st.markdown("**⚠️ Limitations**")
            st.write("The model learns patterns from one survey dataset. It only estimates a score, cannot diagnose anything and never replaces professional advice.")
    st.stop()

# ---------- Tips page ----------
if page == "💡 Mental Health Tips":
    st.markdown('<div class="hero"><h1>Mental Health <span>Tips</span></h1><p class="muted">Small daily habits that support student well-being.</p></div>', unsafe_allow_html=True)
    tips_page = [
        ("😴 Sleep", "Aim for 7 to 9 hours every night and keep a regular sleep time."),
        ("📱 Screen time", "Take a short break every hour and keep your phone away from the bed."),
        ("🏃 Movement", "Even a 20 to 30 minute walk each day improves mood and focus."),
        ("📚 Study routine", "Study in focused blocks with breaks instead of long last-minute sessions."),
        ("🤝 Connection", "Talk to friends or family regularly. Do not carry stress alone."),
        ("🆘 Get help", "If you feel low for many days, speak to a counsellor, doctor or someone you trust."),
    ]
    html = '<div class="grid3">'
    for title, text in tips_page:
        html += f'<div class="panel"><b>{title}</b><br><span class="muted">{text}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.stop()

# ---------- Prediction page ----------
st.markdown('<div class="hero"><h1>Student Mental <span>Health Predictor</span></h1><p class="muted">Estimate a mental health score from lifestyle and social-media habits using a Random Forest model. Nothing you enter is stored.</p></div>', unsafe_allow_html=True)

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

left, right = st.columns([2.4, 1])
values = {}

with left:
    st.markdown('<div class="sec">✦ Enter Your Details</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    values["Age"] = c1.slider("Age", 15, 60, 22)
    values["Gender"] = c2.selectbox("Gender", cat_options["Gender"])
    values["Academic_Level"] = c3.selectbox("Academic Level", cat_options["Academic_Level"])
    values["Stress_Level"] = c4.selectbox("Stress Level", ["Low", "Medium", "High", "Very High"])

    c1, c2, c3, c4 = st.columns(4)
    values["Sleep_Hours_Per_Night"] = c1.slider("Sleep (hours)", 0.0, 12.0, 7.0, 0.5)
    values["Study_Hours"] = c2.slider("Study (hours/day)", 0.0, 12.0, 4.0, 0.5)
    values["Avg_Daily_Usage_Hours"] = c3.slider("Screen Time (hours)", 0.0, 12.0, 5.0, 0.5)
    values["Physical_Activity_Hours"] = c4.slider("Activity (hours)", 0.0, 6.0, 1.0, 0.5)

    c1, c2, c3, c4 = st.columns(4)
    values["Daily_Unlocks"] = c1.number_input("Daily Phone Unlocks", 0, 500, 50, 1)
    values["Most_Used_Platform"] = c2.selectbox("Most Used Platform", cat_options["Most_Used_Platform"])
    values["Purpose_Of_Use"] = c3.selectbox("Purpose of Use", cat_options["Purpose_Of_Use"])
    values["Grouped_country"] = c4.selectbox("Country Group", cat_options["Grouped_country"])

    predict = st.button("✦ Predict Mental Health Score →")
    tips_box = st.empty()  # suggestions or errors are filled in here after the prediction

with right:
    st.markdown('<div class="sec">📈 Prediction Result</div>', unsafe_allow_html=True)

    # Sleep + study + exercise cannot add up to more than 24 hours in a day.
    busy_hours = values["Sleep_Hours_Per_Night"] + values["Study_Hours"] + values["Physical_Activity_Hours"]
    current_band = None
    result_html = """
    <div class="panel"><div class="row">
      <div class="ring" style="background:#1d2a46"><div class="ring-inner"><div class="big">—</div></div></div>
      <div class="rtext"><b>Ready when you are</b><br><span class="muted">Fill in the form and click Predict.</span></div>
    </div></div>
    """

    if predict and busy_hours > 24:
        tips_box.error(f"Sleep, study and physical activity add up to {busy_hours:.1f} hours, which is more than a day. Please correct your inputs.")
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

            result_html = f"""
            <div class="panel"><div class="row">
              <div class="ring" style="--target:{percent*3.6};--p:{percent*3.6};background:conic-gradient({colour} calc(var(--p)*1deg),#1d2a46 0deg);animation:fillRing 1.6s ease-out forwards;box-shadow:0 0 20px {colour}55">
                <div class="ring-inner"><div class="big">{prediction:.1f}</div><div class="muted">/ {MAX_SCORE:.0f}</div></div>
              </div>
              <div class="rtext"><b>Mental Health Score</b>
                <div class="result" style="border:1px solid {colour};background:{colour}22"><b>{face} {label}</b><br><span class="muted">{message}</span></div>
              </div>
            </div></div>
            """

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
            tips_box.markdown(f'<div class="sugg"><b>💗 Suggestions for you</b><ul class="muted">{tips_html}</ul></div>', unsafe_allow_html=True)
        except Exception as e:
            tips_box.error(f"Prediction failed. Details: {e}")

    st.markdown(result_html, unsafe_allow_html=True)

    bands_html = ""
    ranges = ["80 - 100%", "60 - 79%", "40 - 59%", "20 - 39%", "0 - 19%"]
    index = 0
    for band in BANDS:
        css = "band current" if (current_band is not None and band[0] == current_band[0]) else "band"
        bands_html += f'<div class="{css}"><span class="tag" style="background:{band[2]}">{ranges[index]}</span>{band[1]}</div>'
        index += 1
    st.markdown(f'<div class="panel"><b>💡 Score Interpretation</b>{bands_html}</div>', unsafe_allow_html=True)
