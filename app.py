import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Tech Salary Predictor", page_icon="💼", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); min-height: 100vh; }
.main-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 2.5rem; margin-bottom: 1.5rem; }
.header-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7b61ff, #ff6b9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; line-height: 1.2; }
.header-sub { color: rgba(255,255,255,0.45); font-size: 0.95rem; margin-top: 0.4rem; }
.section-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00d4ff; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1rem; margin-top: 0.5rem; }
.result-box { background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,97,255,0.12)); border: 1px solid rgba(0,212,255,0.3); border-radius: 20px; padding: 2.5rem; text-align: center; margin-top: 1rem; }
.result-label { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: rgba(255,255,255,0.45); letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.3rem; }
.result-salary { font-family: 'Space Mono', monospace; font-size: 3.5rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7b61ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0.2rem 0 0.4rem; line-height: 1; }
.result-monthly { color: rgba(255,255,255,0.4); font-size: 0.9rem; margin-bottom: 1.2rem; }
.range-row { display: flex; justify-content: center; gap: 2.5rem; margin: 1rem 0; }
.range-item { text-align: center; }
.range-val { font-family: 'Space Mono', monospace; font-size: 1.05rem; font-weight: 700; color: rgba(255,255,255,0.85); }
.range-lbl { font-size: 0.7rem; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }
.info-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.2rem; justify-content: center; }
.info-pill { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.78rem; color: rgba(255,255,255,0.55); }
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label { color: rgba(255,255,255,0.75) !important; font-size: 0.88rem !important; font-weight: 500 !important; }
div.stButton > button { width: 100%; background: linear-gradient(90deg, #00d4ff, #7b61ff); color: white; font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700; letter-spacing: 0.08em; border: none; border-radius: 12px; padding: 0.85rem 2rem; margin-top: 1rem; cursor: pointer; }
div.stButton > button:hover { opacity: 0.85; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tech_job_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-card">
  <p class="header-title">Tech Salary<br>Predictor</p>
  <p class="header-sub">Enter your details to get an instant salary estimate</p>
</div>
""", unsafe_allow_html=True)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("❌ `tech_job_model.pkl` not found. Run Step 10 in your notebook first, then place the pkl in the same folder as app.py")
    st.stop()

model     = bundle["model"]
scaler    = bundle["scaler"]
le_gender = bundle["le_gender"]
le_edu    = bundle["le_edu"]
le_title  = bundle["le_title"]
TECH_JOBS = bundle["tech_jobs"]
GENDERS   = bundle["genders"]
EDU_LEVELS= bundle["edu_levels"]

# ── Inputs ───────────────────────────────────────────────────
st.markdown('<p class="section-label">Enter Your Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    default_job = TECH_JOBS.index("Software Engineer") if "Software Engineer" in TECH_JOBS else 0
    job_title = st.selectbox("Job Title", TECH_JOBS, index=default_job)
    education = st.selectbox("Education Level", EDU_LEVELS)
    gender    = st.selectbox("Gender", GENDERS)
with col2:
    age       = st.number_input("Age", min_value=18, max_value=70, value=28, step=1)
    years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=3.0, step=0.5)

st.markdown("---")

# ── Predict ──────────────────────────────────────────────────
if st.button("💰  PREDICT MY SALARY"):
    try:
        num_scaled = scaler.transform([[age, years_exp]])[0]
        features   = np.array([[
            num_scaled[0],
            num_scaled[1],
            le_gender.transform([gender])[0],
            le_edu.transform([education])[0],
            le_title.transform([job_title])[0],
        ]])

        salary  = model.predict(features)[0]
        monthly = salary / 12
        low     = max(0, salary - 8000)
        high    = salary + 8000

        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">Estimated Annual Salary</p>
            <p class="result-salary">${salary:,.0f}</p>
            <p class="result-monthly">${monthly:,.0f} per month</p>
            <div class="range-row">
                <div class="range-item">
                    <div class="range-val">${low:,.0f}</div>
                    <div class="range-lbl">Low</div>
                </div>
                <div class="range-item">
                    <div class="range-val">${salary:,.0f}</div>
                    <div class="range-lbl">Predicted</div>
                </div>
                <div class="range-item">
                    <div class="range-val">${high:,.0f}</div>
                    <div class="range-lbl">High</div>
                </div>
            </div>
            <div class="info-row">
                <span class="info-pill">💼 {job_title}</span>
                <span class="info-pill">🎓 {education}</span>
                <span class="info-pill">📅 {years_exp} yrs exp</span>
                <span class="info-pill">🎂 Age {age}</span>
                <span class="info-pill">👤 {gender}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.72rem;font-family:Space Mono,monospace;">Random Forest Regressor · Tech Job Salary Data</p>', unsafe_allow_html=True)