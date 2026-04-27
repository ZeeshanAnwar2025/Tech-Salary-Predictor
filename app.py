"""
╔══════════════════════════════════════════════════════════════╗
║         TECH SALARY PREDICTOR  —  Final Year Project        ║
║  Upgraded: Login · Sidebar Nav · History · Dashboard · PDF  ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pickle
import os
import sqlite3
import hashlib
import io
import csv
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS  — original design tokens preserved + new components
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ─────────────────────────────────────────────── */
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); min-height: 100vh; }

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Sidebar toggle — always visible ─────────────────── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    color: #00d4ff !important;
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="collapsedControl"] svg {
    fill: #00d4ff !important;
    stroke: #00d4ff !important;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1e 0%, #12122a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Original main-card ──────────────────────────────── */
.main-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
}

/* ── Original typography ─────────────────────────────── */
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.2;
}
.header-sub { color: rgba(255,255,255,0.45); font-size: 0.95rem; margin-top: 0.4rem; }
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: #00d4ff;
    letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 1rem; margin-top: 0.5rem;
}

/* ── Original result box (UNTOUCHED) ─────────────────── */
.result-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,97,255,0.12));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px; padding: 2.5rem;
    text-align: center; margin-top: 1rem;
}
.result-label {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: rgba(255,255,255,0.45); letter-spacing: 0.18em;
    text-transform: uppercase; margin-bottom: 0.3rem;
}
.result-salary {
    font-family: 'Space Mono', monospace; font-size: 3.5rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0.2rem 0 0.4rem; line-height: 1;
}
.result-monthly  { color: rgba(255,255,255,0.4); font-size: 0.9rem; margin-bottom: 1.2rem; }
.range-row       { display: flex; justify-content: center; gap: 2.5rem; margin: 1rem 0; }
.range-item      { text-align: center; }
.range-val       { font-family: 'Space Mono', monospace; font-size: 1.05rem; font-weight: 700; color: rgba(255,255,255,0.85); }
.range-lbl       { font-size: 0.7rem; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }
.info-row        { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.2rem; justify-content: center; }
.info-pill       { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.78rem; color: rgba(255,255,255,0.55); }

/* ── Original button (extended with hover) ───────────── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    color: white; font-family: 'Space Mono', monospace;
    font-size: 1rem; font-weight: 700; letter-spacing: 0.08em;
    border: none; border-radius: 12px;
    padding: 0.85rem 2rem; margin-top: 1rem; cursor: pointer;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
}
div.stButton > button:hover {
    opacity: 0.88; transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.25);
}
div.stButton > button:active { transform: translateY(0); }

/* ── Form inputs ─────────────────────────────────────── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.88rem !important; font-weight: 500 !important;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }

/* ── Hero (new) ──────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(123,97,255,0.10), rgba(255,107,157,0.05));
    border: 1px solid rgba(0,212,255,0.18); border-radius: 24px;
    padding: 4rem 2.5rem 3rem; text-align: center; margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Mono', monospace; font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff, #ff6b9d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.6rem; line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem; color: rgba(255,255,255,0.45);
    margin-bottom: 2rem; max-width: 560px; margin-left: auto; margin-right: auto;
}
.hero-badge {
    display: inline-block; background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25); border-radius: 999px;
    padding: 0.25rem 1rem; font-size: 0.75rem; color: #00d4ff;
    font-family: 'Space Mono', monospace; letter-spacing: 0.1em; margin-bottom: 1.5rem;
}

/* ── Feature cards (new) ─────────────────────────────── */
.feat-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.6rem 1.4rem; height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-icon  { font-size: 2rem; margin-bottom: 0.7rem; }
.feat-title { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #00d4ff; margin-bottom: 0.4rem; letter-spacing: 0.05em; }
.feat-desc  { font-size: 0.82rem; color: rgba(255,255,255,0.4); line-height: 1.5; }

/* ── Sidebar nav logo / user chip ────────────────────── */
.nav-logo {
    font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; padding: 0.5rem 0 1.5rem; text-align: center; letter-spacing: 0.05em;
}
.user-chip {
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 999px; padding: 0.4rem 1rem; font-size: 0.78rem;
    color: rgba(255,255,255,0.6); text-align: center; margin-bottom: 1.2rem;
}

/* ── History rows ────────────────────────────────────── */
.history-row {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.5rem;
}
.history-salary {
    font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.history-meta { font-size: 0.78rem; color: rgba(255,255,255,0.35); }

/* ── Login card ──────────────────────────────────────── */
.login-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(0,212,255,0.18);
    border-radius: 24px; padding: 2.8rem 2.2rem;
}
.login-title {
    font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}
.login-sub { color: rgba(255,255,255,0.35); font-size: 0.85rem; margin-bottom: 1.5rem; }

/* ── Stat cards ──────────────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,97,255,0.08));
    border: 1px solid rgba(0,212,255,0.2); border-radius: 14px;
    padding: 1.4rem 1.2rem; text-align: center;
}
.stat-val { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #00d4ff; }
.stat-lbl { font-size: 0.75rem; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }

/* ── Alerts ──────────────────────────────────────────── */
.success-alert {
    background: rgba(0,212,100,0.1); border: 1px solid rgba(0,212,100,0.3);
    border-radius: 10px; padding: 0.8rem 1.2rem;
    color: #00d464; font-size: 0.88rem; margin-top: 0.8rem;
}
.info-alert {
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.25);
    border-radius: 10px; padding: 0.8rem 1.2rem;
    color: #00d4ff; font-size: 0.88rem; margin-top: 0.8rem;
}

/* ── Page headers ────────────────────────────────────── */
.page-title {
    font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}
.page-sub { color: rgba(255,255,255,0.35); font-size: 0.85rem; margin-bottom: 1.5rem; }

/* ── Download button ─────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(90deg, #00d4ff, #7b61ff) !important;
    color: white !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    opacity: 0.85 !important; transform: translateY(-2px) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stTabs"] button {
    color: rgba(255,255,255,0.45) !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* ── Footer ──────────────────────────────────────────── */
.footer {
    text-align: center; color: rgba(255,255,255,0.15);
    font-size: 0.72rem; font-family: 'Space Mono', monospace;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="DM Sans", color="rgba(255,255,255,0.65)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=20, r=20, t=40, b=20),
)
GRAD_COLORS = ["#00d4ff", "#7b61ff", "#ff6b9d", "#00ff9d", "#ffb347"]
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salary_app.db")


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# ─── SQLite setup ────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, password TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL, job_title TEXT, education TEXT,
        gender TEXT, age INTEGER, experience REAL,
        predicted_salary REAL, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, name TEXT, rating INTEGER,
        comment TEXT, created_at TEXT NOT NULL)""")
    conn.commit()
    return conn


def db_register(username, password):
    conn = get_db()
    try:
        conn.execute("INSERT INTO users VALUES (?,?,?)",
                     (username.strip().lower(), hash_pw(password), datetime.now().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def db_login(username, password):
    conn = get_db()
    row = conn.execute("SELECT 1 FROM users WHERE username=? AND password=?",
                       (username.strip().lower(), hash_pw(password))).fetchone()
    conn.close()
    return row is not None


def db_save_prediction(username, job_title, education, gender, age, experience, salary):
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (username,job_title,education,gender,age,experience,predicted_salary,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (username, job_title, education, gender, age, experience, salary,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def db_get_history(username):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE username=? ORDER BY id DESC LIMIT 50", (username,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_all_predictions():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_save_feedback(username, name, rating, comment):
    conn = get_db()
    conn.execute("INSERT INTO feedback (username,name,rating,comment,created_at) VALUES (?,?,?,?,?)",
                 (username, name, rating, comment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


# ─── Model loader (original logic, unchanged) ────────────────
@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tech_job_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Session state defaults ──────────────────────────────────
for _k, _v in {
    "logged_in": False, "username": "",
    "page": "🏠  Home", "last_prediction": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════
#  LOGIN / SIGNUP
# ═══════════════════════════════════════════════════════════════
def page_login():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 0 1.5rem;">
          <div style="font-family:'Space Mono',monospace;font-size:1.9rem;font-weight:700;
               background:linear-gradient(90deg,#00d4ff,#7b61ff,#ff6b9d);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;line-height:1.2;">Tech Salary<br>Predictor</div>
          <div style="color:rgba(255,255,255,0.28);font-size:0.78rem;
               font-family:'Space Mono',monospace;letter-spacing:0.12em;margin-top:0.6rem;">
               AI-POWERED · MINOR PROJECT
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["  Sign In  ", "  Create Account  "])

        with tab_in:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown('<p class="login-title">Welcome back</p>', unsafe_allow_html=True)
            st.markdown('<p class="login-sub">Sign in to continue</p>', unsafe_allow_html=True)
            u = st.text_input("Username", key="li_u", placeholder="your_username")
            p = st.text_input("Password", type="password", key="li_p", placeholder="••••••••")
            if st.button("Sign In →", key="btn_li"):
                if not u or not p:
                    st.warning("Fill in both fields.")
                elif db_login(u, p):
                    st.session_state.logged_in = True
                    st.session_state.username  = u.strip().lower()
                    st.session_state.page      = "🏠  Home"
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try again.")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_up:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown('<p class="login-title">Create account</p>', unsafe_allow_html=True)
            st.markdown('<p class="login-sub">Join to start predicting</p>', unsafe_allow_html=True)
            nu  = st.text_input("Username",         key="su_u",  placeholder="choose_username")
            np1 = st.text_input("Password",         type="password", key="su_p1", placeholder="min 6 chars")
            np2 = st.text_input("Confirm password", type="password", key="su_p2", placeholder="repeat")
            if st.button("Create Account →", key="btn_su"):
                if not nu or not np1 or not np2:
                    st.warning("Fill in all fields.")
                elif len(np1) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif np1 != np2:
                    st.error("Passwords do not match.")
                elif db_register(nu, np1):
                    st.success("Account created! Sign in above.")
                else:
                    st.error("Username already taken.")
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
PAGES = [
    "🏠  Home",
    "💰  Predict Salary",
    "📋  Prediction History",
    "📊  Dashboard",
    "📄  Download Report",
    "🤖  About Model",
    "💬  Feedback",
]
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="nav-logo">⚡ SALARY.AI</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="user-chip">👤 &nbsp;{st.session_state.username}</div>',
            unsafe_allow_html=True,
        )
        for pg in PAGES:
            active = st.session_state.page == pg
            if active:
                st.markdown(
                    f'<div style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);'
                    f'color:#00d4ff;border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.25rem;'
                    f'font-size:0.9rem;">{pg}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(pg, key=f"nav_{pg}", use_container_width=True):
                    st.session_state.page = pg
                    st.rerun()

        st.markdown("---")
        if st.button("🚪  Logout", key="btn_logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown(
            '<div class="footer">Random Forest Regressor<br>© 2025 Tech Salary Predictor</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
#  PAGE — HOME
# ═══════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🚀 AI-POWERED SALARY INTELLIGENCE</div>
        <div class="hero-title">Predict Your Tech<br>Salary Instantly</div>
        <div class="hero-sub">
            Enter your job title, education, and experience to receive a
            data-driven salary estimate powered by a trained
            Random Forest machine learning model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_cta, _, _ = st.columns([1, 1, 1])
    with c_cta:
        if st.button("💰  Predict My Salary →", key="hero_cta"):
            st.session_state.page = "💰  Predict Salary"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">What You Get</p>', unsafe_allow_html=True)

    features = [
        ("🤖", "ML Predictions",   "Random Forest trained on real tech salary data for high accuracy estimates."),
        ("📊", "Live Dashboard",    "Interactive Plotly charts: salary by role, trends, education, experience."),
        ("📋", "Full History",      "Every prediction saved with timestamp so you can track your data over time."),
        ("📄", "Downloadable Report","Export a professional salary report as a text or CSV file instantly."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(f"""
        <div class="feat-card">
            <div class="feat-icon">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    all_preds = db_all_predictions()
    if all_preds:
        st.markdown('<p class="section-label">Live Stats</p>', unsafe_allow_html=True)
        df_all = pd.DataFrame(all_preds)
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(f'<div class="stat-card"><div class="stat-val">{len(df_all)}</div><div class="stat-lbl">Predictions Made</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-card"><div class="stat-val">${df_all["predicted_salary"].mean():,.0f}</div><div class="stat-lbl">Avg Salary</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-card"><div class="stat-val">${df_all["predicted_salary"].max():,.0f}</div><div class="stat-lbl">Highest Prediction</div></div>', unsafe_allow_html=True)
        s4.markdown(f'<div class="stat-card"><div class="stat-val">{df_all["job_title"].nunique()}</div><div class="stat-lbl">Unique Roles</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Random Forest Regressor · Tech Job Salary Dataset · Minor Project 2026</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — PREDICT SALARY  (original logic fully preserved)
# ═══════════════════════════════════════════════════════════════
def page_predict(bundle):
    model      = bundle["model"]
    scaler     = bundle["scaler"]
    le_gender  = bundle["le_gender"]
    le_edu     = bundle["le_edu"]
    le_title   = bundle["le_title"]
    TECH_JOBS  = bundle["tech_jobs"]
    GENDERS    = bundle["genders"]
    EDU_LEVELS = bundle["edu_levels"]

    # Original header card
    st.markdown("""
    <div class="main-card">
      <p class="header-title">Tech Salary<br>Predictor</p>
      <p class="header-sub">Enter your details to get an instant salary estimate</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Enter Your Details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        default_job = TECH_JOBS.index("Software Engineer") if "Software Engineer" in TECH_JOBS else 0
        job_title = st.selectbox("Job Title",       TECH_JOBS, index=default_job)
        education = st.selectbox("Education Level", EDU_LEVELS)
        gender    = st.selectbox("Gender",          GENDERS)
    with col2:
        age       = st.number_input("Age",                    min_value=18,  max_value=70,  value=28,  step=1)
        years_exp = st.number_input("Years of Experience",    min_value=0.0, max_value=40.0, value=3.0, step=0.5)

    st.markdown("---")

    if st.button("💰  PREDICT MY SALARY"):
        with st.spinner("Running prediction model…"):
            try:
                num_scaled = scaler.transform([[age, years_exp]])[0]
                features   = np.array([[
                    num_scaled[0], num_scaled[1],
                    le_gender.transform([gender])[0],
                    le_edu.transform([education])[0],
                    le_title.transform([job_title])[0],
                ]])

                salary  = model.predict(features)[0]
                monthly = salary / 12
                low     = max(0, salary - 8000)
                high    = salary + 8000

                db_save_prediction(
                    st.session_state.username,
                    job_title, education, gender, int(age), float(years_exp), float(salary)
                )
                st.session_state.last_prediction = {
                    "job_title": job_title, "education": education, "gender": gender,
                    "age": int(age), "experience": float(years_exp), "salary": float(salary),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "username": st.session_state.username,
                }

                # Original result card — untouched
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

                st.markdown(
                    '<div class="success-alert">✅ Prediction saved to your history.</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.72rem;'
        'font-family:Space Mono,monospace;">Random Forest Regressor · Tech Job Salary Data</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE — PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<p class="page-title">Prediction History</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">All salary predictions saved to your account</p>', unsafe_allow_html=True)

    rows = db_get_history(st.session_state.username)
    if not rows:
        st.markdown(
            '<div class="info-alert">📭 No predictions yet. Head to <b>Predict Salary</b> to get started.</div>',
            unsafe_allow_html=True,
        )
        return

    df_h = pd.DataFrame(rows)
    st.download_button(
        "⬇️  Export History as CSV",
        data=df_h.to_csv(index=False).encode(),
        file_name=f"history_{st.session_state.username}.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    for r in rows:
        st.markdown(f"""
        <div class="history-row">
            <div>
                <div style="font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.8);">
                    💼 {r['job_title']}
                </div>
                <div class="history-meta">
                    🎓 {r['education']} &nbsp;|&nbsp; 📅 {r['experience']} yrs
                    &nbsp;|&nbsp; 🎂 Age {r['age']} &nbsp;|&nbsp; 👤 {r['gender']}
                </div>
                <div class="history-meta" style="margin-top:0.2rem;">🕒 {r['created_at']}</div>
            </div>
            <div class="history-salary">${r['predicted_salary']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<p class="page-title">Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Visual insights from all predictions in the system</p>', unsafe_allow_html=True)

    rows = db_all_predictions()
    if len(rows) < 3:
        st.markdown(
            '<div class="info-alert">📊 Not enough data yet — make a few predictions to unlock the dashboard.</div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(rows)

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label">Avg Salary by Job Role</p>', unsafe_allow_html=True)
        role_avg = df.groupby("job_title")["predicted_salary"].mean().sort_values(ascending=True).reset_index()
        role_avg.columns = ["Job Title", "Avg Salary"]
        fig = go.Figure(go.Bar(
            x=role_avg["Avg Salary"], y=role_avg["Job Title"], orientation="h",
            marker=dict(color=role_avg["Avg Salary"], colorscale=[[0,"#7b61ff"],[1,"#00d4ff"]], showscale=False),
            text=[f"${v:,.0f}" for v in role_avg["Avg Salary"]],
            textposition="outside", textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=max(280, len(role_avg)*34))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-label">Salary Distribution</p>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Histogram(
            x=df["predicted_salary"], nbinsx=20,
            marker=dict(color="#7b61ff", line=dict(color="rgba(0,212,255,0.4)", width=1)),
            opacity=0.85,
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title="Salary ($)", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-label">Experience vs Salary</p>', unsafe_allow_html=True)
        fig3 = px.scatter(
            df, x="experience", y="predicted_salary", color="job_title",
            color_discrete_sequence=GRAD_COLORS, opacity=0.75,
            labels={"experience": "Years of Experience", "predicted_salary": "Salary ($)", "job_title": "Role"},
        )
        if len(df) >= 2:
            m, b = np.polyfit(df["experience"], df["predicted_salary"], 1)
            tx = np.linspace(df["experience"].min(), df["experience"].max(), 100)
            fig3.add_trace(go.Scatter(x=tx, y=m*tx+b, mode="lines",
                                      line=dict(color="#00d4ff", width=2, dash="dot"), name="Trend"))
        fig3.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<p class="section-label">Education vs Avg Salary</p>', unsafe_allow_html=True)
        edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
        edu_avg = df.groupby("education")["predicted_salary"].mean().reindex(edu_order).dropna().reset_index()
        edu_avg.columns = ["Education", "Avg Salary"]
        fig4 = go.Figure(go.Bar(
            x=edu_avg["Education"], y=edu_avg["Avg Salary"],
            marker=dict(color=edu_avg["Avg Salary"], colorscale=[[0,"#7b61ff"],[1,"#ff6b9d"]], showscale=False),
            text=[f"${v:,.0f}" for v in edu_avg["Avg Salary"]],
            textposition="outside", textfont=dict(size=11, color="rgba(255,255,255,0.6)"),
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Avg Salary ($)")
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3 — Trend over time
    st.markdown('<p class="section-label">Prediction Trend Over Time</p>', unsafe_allow_html=True)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df_s = df.sort_values("created_at")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df_s["created_at"], y=df_s["predicted_salary"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(color="#7b61ff", size=6),
        fill="tozeroy", fillcolor="rgba(123,97,255,0.08)", name="Salary",
    ))
    fig5.update_layout(**PLOTLY_LAYOUT, height=260, xaxis_title="Date", yaxis_title="Salary ($)")
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — DOWNLOAD REPORT
# ═══════════════════════════════════════════════════════════════
def page_report():
    st.markdown('<p class="page-title">Download Report</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Generate a professional salary prediction report</p>', unsafe_allow_html=True)

    pred = st.session_state.get("last_prediction")
    if not pred:
        rows = db_get_history(st.session_state.username)
        if rows:
            r = rows[0]
            pred = {"username": r["username"], "job_title": r["job_title"],
                    "education": r["education"], "gender": r["gender"],
                    "age": r["age"], "experience": r["experience"],
                    "salary": r["predicted_salary"], "date": r["created_at"]}

    if not pred:
        st.markdown(
            '<div class="info-alert">📭 No prediction found. Make a prediction on the <b>Predict Salary</b> page first.</div>',
            unsafe_allow_html=True,
        )
        return

    low  = max(0, pred["salary"] - 8000)
    high = pred["salary"] + 8000

    # Preview
    st.markdown(f"""
    <div class="result-box" style="text-align:left;max-width:600px;">
        <p class="result-label">Report Preview</p>
        <p class="result-salary" style="font-size:2.5rem;">${pred['salary']:,.0f}</p>
        <p class="result-monthly">${pred['salary']/12:,.0f} / month</p>
        <hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;
                    font-size:0.85rem;color:rgba(255,255,255,0.6);">
            <div>👤 <b>User</b>: {pred['username']}</div>
            <div>💼 <b>Role</b>: {pred['job_title']}</div>
            <div>🎓 <b>Education</b>: {pred['education']}</div>
            <div>📅 <b>Experience</b>: {pred['experience']} yrs</div>
            <div>🎂 <b>Age</b>: {pred['age']}</div>
            <div>👤 <b>Gender</b>: {pred['gender']}</div>
            <div style="grid-column:span 2;">🕒 <b>Date</b>: {pred['date']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    txt_report = f"""
╔══════════════════════════════════════════════════════════════════╗
║             TECH SALARY PREDICTOR — SALARY REPORT               ║
╚══════════════════════════════════════════════════════════════════╝

Generated  : {pred['date']}
Report for : {pred['username']}

──────────────────────────────────────────────────────────────────
  EMPLOYEE DETAILS
──────────────────────────────────────────────────────────────────
  Job Title           : {pred['job_title']}
  Education Level     : {pred['education']}
  Gender              : {pred['gender']}
  Age                 : {pred['age']}
  Years of Experience : {pred['experience']}

──────────────────────────────────────────────────────────────────
  SALARY PREDICTION RESULT
──────────────────────────────────────────────────────────────────
  Predicted Annual Salary : ${pred['salary']:,.0f}
  Monthly Equivalent      : ${pred['salary']/12:,.0f}
  Salary Range (Low)      : ${low:,.0f}
  Salary Range (High)     : ${high:,.0f}

──────────────────────────────────────────────────────────────────
  MODEL INFORMATION
──────────────────────────────────────────────────────────────────
  Algorithm              : Random Forest Regressor
  Dataset                : Tech Job Salary Dataset
  Features Used          : Job Title, Education, Gender,
                           Age, Years of Experience
  Hyperparameters        : n_estimators=200, max_depth=15,
                           min_samples_leaf=3

──────────────────────────────────────────────────────────────────
  DISCLAIMER
──────────────────────────────────────────────────────────────────
  This report is generated by an AI model for educational purposes.
  Actual salaries vary based on company, location, and market
  conditions. Use this estimate as a reference point only.

══════════════════════════════════════════════════════════════════
  Tech Salary Predictor · Final Year Project 2025
══════════════════════════════════════════════════════════════════
""".strip()

    csv_report = (
        "Field,Value\n" +
        "\n".join([
            f"Username,{pred['username']}",
            f"Job Title,{pred['job_title']}",
            f"Education,{pred['education']}",
            f"Gender,{pred['gender']}",
            f"Age,{pred['age']}",
            f"Experience (yrs),{pred['experience']}",
            f"Predicted Annual Salary,${pred['salary']:,.0f}",
            f"Monthly Salary,${pred['salary']/12:,.0f}",
            f"Salary Low,${low:,.0f}",
            f"Salary High,${high:,.0f}",
            f"Report Date,{pred['date']}",
        ])
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2, _ = st.columns([1, 1, 1])
    with col1:
        st.download_button("📄  Download TXT Report",
                           data=txt_report.encode("utf-8"),
                           file_name=f"salary_report_{pred['username']}_{ts}.txt",
                           mime="text/plain")
    with col2:
        st.download_button("📊  Download CSV Report",
                           data=csv_report.encode("utf-8"),
                           file_name=f"salary_report_{pred['username']}_{ts}.csv",
                           mime="text/csv")


# ═══════════════════════════════════════════════════════════════
#  PAGE — ABOUT MODEL
# ═══════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<p class="page-title">About the Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">How the AI salary predictor was built and evaluated</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown('<div class="stat-card"><div class="stat-val">RF</div><div class="stat-lbl">Algorithm</div></div>', unsafe_allow_html=True)
    s2.markdown('<div class="stat-card"><div class="stat-val">5</div><div class="stat-lbl">Input Features</div></div>', unsafe_allow_html=True)
    s3.markdown('<div class="stat-card"><div class="stat-val">200</div><div class="stat-lbl">Trees in Forest</div></div>', unsafe_allow_html=True)
    s4.markdown('<div class="stat-card"><div class="stat-val">80/20</div><div class="stat-lbl">Train / Test Split</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="main-card">
            <p class="section-label">Dataset</p>
            <p style="color:rgba(255,255,255,0.65);font-size:0.9rem;line-height:1.7;">
                Trained on a curated <b style="color:#00d4ff;">Tech Job Salary Dataset</b>
                with thousands of records from real software, data, and engineering roles.
                Only <b style="color:#7b61ff;">tech-specific positions</b> were retained —
                non-tech roles like HR, Sales, and Finance were removed to ensure
                high-quality domain-specific predictions.
            </p>
            <br>
            <p class="section-label">Input Features</p>
            <div style="display:flex;flex-direction:column;gap:0.45rem;">
                <div class="info-pill" style="text-align:left;">💼 &nbsp;Job Title (Label Encoded)</div>
                <div class="info-pill" style="text-align:left;">🎓 &nbsp;Education Level (Ordinal Encoded)</div>
                <div class="info-pill" style="text-align:left;">👤 &nbsp;Gender (Label Encoded)</div>
                <div class="info-pill" style="text-align:left;">🎂 &nbsp;Age (Standard Scaled)</div>
                <div class="info-pill" style="text-align:left;">📅 &nbsp;Years of Experience (Standard Scaled)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="main-card">
            <p class="section-label">Algorithm — Random Forest Regressor</p>
            <p style="color:rgba(255,255,255,0.65);font-size:0.9rem;line-height:1.7;">
                A <b style="color:#00d4ff;">Random Forest</b> builds multiple decision trees
                and averages their outputs for a robust, accurate salary regression prediction.
            </p>
            <br>
            <p class="section-label">Why Random Forest?</p>
            <ul style="color:rgba(255,255,255,0.5);font-size:0.87rem;line-height:2;padding-left:1.2rem;">
                <li>Handles non-linear salary relationships well</li>
                <li>Resistant to overfitting via tree averaging</li>
                <li>Works with mixed numeric + categorical features</li>
                <li>Provides built-in feature importance ranking</li>
                <li>No distribution assumptions required</li>
            </ul>
            <br>
            <p class="section-label">Hyperparameters</p>
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                        color:rgba(0,212,255,0.7);line-height:1.9;">
                n_estimators &nbsp;= 200<br>
                max_depth &nbsp;&nbsp;&nbsp;&nbsp;= 15<br>
                min_samples_leaf = 3<br>
                random_state &nbsp;= 42
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline visualisation
    st.markdown('<p class="section-label">Model Pipeline</p>', unsafe_allow_html=True)
    steps  = ["Raw Dataset", "Filter Tech Roles", "Encode Features", "Scale Numerics", "Train RF", "Predict Salary"]
    colors = ["#7b61ff", "#6a7fff", "#5a9fff", "#3ab8ff", "#18ccff", "#00d4ff"]
    fig = go.Figure()
    for i, (step, col) in enumerate(zip(steps, colors)):
        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=52, color=col, line=dict(color="rgba(255,255,255,0.15)", width=2)),
            text=[f"<b>{i+1}</b>"], textfont=dict(color="white", size=14),
            textposition="middle center",
            hovertemplate=f"<b>{step}</b><extra></extra>", showlegend=False,
        ))
        fig.add_annotation(x=i, y=-0.22, text=step, showarrow=False,
                           font=dict(size=9.5, color="rgba(255,255,255,0.45)"))
        if i < len(steps)-1:
            fig.add_annotation(x=i+0.5, y=0, text="→", showarrow=False,
                               font=dict(size=20, color="rgba(255,255,255,0.2)"))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=155,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, len(steps)-0.5]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.45, 0.3]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — FEEDBACK
# ═══════════════════════════════════════════════════════════════
def page_feedback():
    st.markdown('<p class="page-title">Feedback</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Help us improve the Tech Salary Predictor</p>', unsafe_allow_html=True)

    col_f, col_info = st.columns([2, 1])
    with col_f:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Share Your Experience</p>', unsafe_allow_html=True)
        name    = st.text_input("Your Name", placeholder="e.g. Alex Johnson")
        rating  = st.select_slider("Rating", options=[1,2,3,4,5], value=5,
                                   format_func=lambda x: "⭐"*x)
        comment = st.text_area("Comments / Suggestions",
                               placeholder="What did you like? What can be improved?", height=130)
        if st.button("📨  Submit Feedback", key="btn_fb"):
            if not name.strip():
                st.warning("Please enter your name.")
            elif not comment.strip():
                st.warning("Please leave a comment.")
            else:
                db_save_feedback(st.session_state.username, name.strip(), int(rating), comment.strip())
                st.markdown(
                    '<div class="success-alert">🎉 Thank you for your feedback! It means a lot.</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="main-card" style="text-align:center;">
            <p class="section-label">Why Feedback?</p>
            <div style="font-size:2.5rem;margin:0.8rem 0;">💬</div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.4);line-height:1.7;">
                Your feedback directly shapes model improvements, UI enhancements,
                and the overall experience of this final year project.
            </div>
            <hr style="border-color:rgba(255,255,255,0.07);margin:1.2rem 0;">
            <div class="stat-val" style="font-size:1.2rem;">⭐ ⭐ ⭐ ⭐ ⭐</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:0.3rem;">
                We aim for 5 stars
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ═══════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        page_login()
        return

    render_sidebar()

    bundle = None
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        pass

    page = st.session_state.page

    if page == "🏠  Home":
        page_home()
    elif page == "💰  Predict Salary":
        if bundle is None:
            st.error("❌ `tech_job_model.pkl` not found. "
                     "Train the model and place the pkl file in the same folder as app.py.")
        else:
            page_predict(bundle)
    elif page == "📋  Prediction History":
        page_history()
    elif page == "📊  Dashboard":
        page_dashboard()
    elif page == "📄  Download Report":
        page_report()
    elif page == "🤖  About Model":
        page_about()
    elif page == "💬  Feedback":
        page_feedback()


if __name__ == "__main__":
    main()
