import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import io
import json
import base64
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, ImageFilter
from fpdf import FPDF
import tempfile
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NPDR Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS – clinical dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-base:       #05080f;
    --bg-surface:    #070d18;
    --bg-card:       rgba(6, 14, 26, 0.88);
    --bg-glass:      rgba(8, 18, 34, 0.65);
    --accent-teal:   #00d4aa;
    --accent-cyan:   #00b4d8;
    --accent-gold:   #f0c060;
    --accent-rose:   #ff6b8a;
    --text-primary:  #e8f4f0;
    --text-secondary:#7aa89a;
    --text-muted:    #3a6458;
    --border:        rgba(0, 212, 170, 0.12);
    --border-bright: rgba(0, 212, 170, 0.35);
    --shadow:        0 24px 64px rgba(0,0,0,0.7);
    --glow-teal:     0 0 40px rgba(0, 212, 170, 0.15);
    --glow-cyan:     0 0 30px rgba(0, 180, 216, 0.2);
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
    --radius:        14px;
    --radius-sm:     8px;
}

/* ── ANIMATED RETINAL BACKGROUND ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-base);
    color: var(--text-primary);
}

.stApp {
    background:
        radial-gradient(ellipse 55% 45% at 15% 15%, rgba(0,180,130,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 85% 80%, rgba(0,140,200,0.08) 0%, transparent 50%),
        radial-gradient(ellipse 70% 60% at 50% 100%, rgba(0,100,80,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 120% 100% at 50% 50%, #060c18 0%, #030810 60%, #020609 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* Retinal vessel-like grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse 2px 60px at 20% 30%, rgba(0,212,170,0.04) 0%, transparent 100%),
        radial-gradient(ellipse 2px 40px at 70% 60%, rgba(0,180,216,0.03) 0%, transparent 100%),
        linear-gradient(rgba(0,212,170,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.018) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 80px 80px, 80px 80px;
    pointer-events: none;
    z-index: 0;
}

/* Floating orbs animation */
@keyframes floatOrb1 {
    0%,100% { transform: translate(0,0) scale(1); opacity:0.6; }
    50%      { transform: translate(30px,-20px) scale(1.1); opacity:0.9; }
}
@keyframes floatOrb2 {
    0%,100% { transform: translate(0,0) scale(1); opacity:0.5; }
    50%      { transform: translate(-20px,25px) scale(0.95); opacity:0.8; }
}
.stApp::after {
    content: '';
    position: fixed;
    top: 10%; right: 8%;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,170,0.05) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
    animation: floatOrb1 12s ease-in-out infinite;
}

/* ── RETINAL EYE LOGO ── */
.retinal-logo-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 6px;
}
.retinal-logo-wrap svg {
    filter: drop-shadow(0 0 12px rgba(0,212,170,0.5));
    flex-shrink: 0;
}
.logo-text-block { text-align: left; }
.logo-title {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 60%, #7dd4f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    letter-spacing: -1px;
}
.logo-sub-text {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Centered login logo variant */
.login-logo-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.login-logo-wrap svg {
    filter: drop-shadow(0 0 18px rgba(0,212,170,0.6));
}
.login-logo {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
    letter-spacing: -1px;
}
.login-sub {
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 32px;
}

/* ── SIGN IN BUTTON — FULL COLOR ── */
[data-testid="stForm"] .stButton > button,
[data-testid="stForm"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, #00c49a 0%, #0096c7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 24px rgba(0,180,150,0.35) !important;
    transition: all 0.25s !important;
    width: 100% !important;
}
[data-testid="stForm"] .stButton > button:hover,
[data-testid="stForm"] .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #00ddb0 0%, #00aadf 100%) !important;
    box-shadow: 0 6px 32px rgba(0,200,160,0.5), 0 0 0 1px rgba(0,212,170,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040a14 0%, #030810 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 280px;
    background: radial-gradient(ellipse at top, rgba(0,212,170,0.06) 0%, transparent 70%);
    pointer-events: none;
}

/* ── MAIN HEADER ── */
.main-header {
    background: linear-gradient(135deg,
        rgba(0,28,20,0.92) 0%,
        rgba(0,38,28,0.85) 50%,
        rgba(0,22,18,0.92) 100%);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius);
    padding: 28px 36px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow-teal), inset 0 1px 0 rgba(0,212,170,0.1);
}
.main-header::before {
    content: '';
    position: absolute;
    top: -60%; right: -5%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,170,0.5), transparent);
}
.main-header h1 {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 60%, #7dd4f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: var(--text-secondary);
    font-size: 0.88rem;
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.badge {
    display: inline-block;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    color: var(--accent-teal);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 18px 20px;
    margin-bottom: 12px;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    border-color: rgba(0,212,170,0.25);
    box-shadow: 0 4px 20px rgba(0,212,170,0.08);
}
.metric-card h4 {
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0 0 8px 0;
    font-weight: 500;
}
.metric-card .value {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-teal);
}
.metric-card .unit {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-left: 4px;
}

/* ── SEVERITY BADGES ── */
.severity-no-dr    { background:rgba(0,212,100,0.08); border:1px solid rgba(0,212,100,0.3); color:#00d464; }
.severity-mild     { background:rgba(160,220,80,0.08); border:1px solid rgba(160,220,80,0.3); color:#a8e066; }
.severity-moderate { background:rgba(240,192,60,0.08); border:1px solid rgba(240,192,60,0.3); color:#f0c040; }
.severity-severe   { background:rgba(255,120,50,0.08); border:1px solid rgba(255,120,50,0.3); color:#ff7832; }
.severity-prolif   { background:rgba(255,80,100,0.08); border:1px solid rgba(255,80,100,0.3); color:#ff5064; }

.severity-badge {
    display: inline-block;
    border-radius: var(--radius-sm);
    padding: 10px 22px;
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 500;
    margin: 12px 0;
    letter-spacing: 0.5px;
    backdrop-filter: blur(8px);
}

/* ── SECTION TITLES ── */
.section-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin: 28px 0 16px 0;
    position: relative;
}
.section-title::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 40px; height: 1px;
    background: var(--accent-teal);
}

/* ── MEDICAL BOX ── */
.med-box {
    background: rgba(0,212,170,0.03);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-teal);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 18px 22px;
    font-size: 0.87rem;
    line-height: 1.8;
    color: #9abfb8;
    backdrop-filter: blur(8px);
}
.med-box .finding {
    color: var(--accent-teal);
    font-weight: 500;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── HISTORY ROWS ── */
.hist-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    background: rgba(4,22,32,0.7);
    border: 1px solid var(--border);
    font-size: 0.83rem;
    backdrop-filter: blur(8px);
    transition: all 0.2s;
}
.hist-row:hover {
    background: rgba(0,212,170,0.04);
    border-color: rgba(0,212,170,0.2);
    transform: translateX(2px);
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(0,180,216,0.08));
    color: var(--accent-teal);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 1px;
    padding: 10px 24px;
    transition: all 0.25s;
    text-transform: uppercase;
    backdrop-filter: blur(8px);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c49a, #0096c7);
    border-color: transparent;
    box-shadow: 0 4px 20px rgba(0,196,154,0.35);
    transform: translateY(-1px);
    color: #fff;
}

div[data-testid="stFileUploader"] {
    background: rgba(0,212,170,0.02);
    border: 2px dashed rgba(0,212,170,0.2);
    border-radius: var(--radius);
    padding: 16px;
    transition: all 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,170,0.45);
    background: rgba(0,212,170,0.04);
    box-shadow: var(--glow-teal);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(4,12,22,0.85);
    border-radius: var(--radius-sm);
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.73rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 6px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-teal) !important;
    background: rgba(0,212,170,0.1) !important;
    border-radius: 6px;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {
    background: rgba(4,14,24,0.85) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0,212,170,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.1) !important;
}

.stSlider > div > div > div > div { background: var(--accent-teal) !important; }
.stAlert { border-radius: var(--radius-sm); }
hr { border-color: var(--border); margin: 20px 0; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-teal); }

/* ── LOGIN CARD ── */
.login-card {
    background: rgba(5,14,26,0.88);
    border: 1px solid var(--border-bright);
    border-radius: 20px;
    padding: 48px 44px 40px 44px;
    width: 100%;
    max-width: 440px;
    margin: 30px auto;
    box-shadow: var(--shadow), var(--glow-teal);
    text-align: center;
    backdrop-filter: blur(28px);
    position: relative;
    overflow: hidden;
}
.login-card::before {
    content: '';
    position: absolute;
    top: -100px; left: 50%;
    transform: translateX(-50%);
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,212,170,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.login-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 150px; height: 150px;
    background: radial-gradient(circle at bottom right, rgba(0,180,216,0.06) 0%, transparent 70%);
    pointer-events: none;
}

/* Animations */
@keyframes pulse-teal {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.75; }
}
.pulse { animation: pulse-teal 2.5s ease-in-out infinite; }

@keyframes rotateEye {
    0%   { transform: scale(1) rotate(0deg); }
    50%  { transform: scale(1.04) rotate(2deg); }
    100% { transform: scale(1) rotate(0deg); }
}
.eye-logo-anim { animation: rotateEye 6s ease-in-out infinite; }

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stTabs { animation: fadeSlideUp 0.5s ease forwards; }
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "current_username" not in st.session_state:
    st.session_state.current_username = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"

# ─────────────────────────────────────────────
# SQLite USER DATABASE
# ─────────────────────────────────────────────
import hashlib
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npdr_users.db")

def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create users table if it doesn't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                full_name   TEXT    NOT NULL,
                password    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            )
        """)
        conn.commit()

# Initialise DB on startup
init_db()

def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def do_register(full_name: str, username: str, password: str) -> tuple:
    """Returns (success: bool, message: str)"""
    full_name = full_name.strip()
    username  = username.strip().lower()
    if not full_name or not username or not password:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, full_name, password, created_at) VALUES (?, ?, ?, ?)",
                (username, full_name, _hash_pw(password),
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
        return True, f"Account created! Welcome, {full_name}."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    except Exception as e:
        return False, f"Registration failed: {e}"

def do_login(username: str, password: str) -> bool:
    u = username.strip().lower()
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT full_name, password FROM users WHERE username = ?", (u,)
            ).fetchone()
        if row and row["password"] == _hash_pw(password):
            st.session_state.logged_in        = True
            st.session_state.current_user     = row["full_name"]
            st.session_state.current_username = u
            return True
    except Exception:
        pass
    return False

def do_logout():
    st.session_state.logged_in        = False
    st.session_state.current_user     = None
    st.session_state.current_username = None
    st.session_state.history          = []
    st.session_state.total_scans      = 0
    st.session_state.auth_page        = "login"

def get_all_users() -> list:
    """Return list of all registered users (for admin view)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, full_name, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

# ─────────────────────────────────────────────
# AUTH PAGE  (Login / Register)
# ─────────────────────────────────────────────
if not st.session_state.logged_in:

    col_l, col_mid, col_r = st.columns([1, 1.2, 1])
    with col_mid:

        # Logo
        st.markdown("""
        <div style="text-align:center;padding:20px 0 4px 0;">
            <div class="login-logo-wrap">
                <svg class="eye-logo-anim" width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <!-- Outer eye shape -->
                  <ellipse cx="36" cy="36" rx="34" ry="34" fill="none" stroke="rgba(0,212,170,0.18)" stroke-width="1"/>
                  <!-- Eye outline -->
                  <path d="M6 36 Q20 14 36 14 Q52 14 66 36 Q52 58 36 58 Q20 58 6 36Z"
                        fill="rgba(0,212,170,0.06)" stroke="#00d4aa" stroke-width="1.4"/>
                  <!-- Retinal vessels radiating from center -->
                  <line x1="36" y1="36" x2="24" y2="22" stroke="rgba(0,212,170,0.35)" stroke-width="0.8"/>
                  <line x1="36" y1="36" x2="20" y2="34" stroke="rgba(0,212,170,0.3)"  stroke-width="0.8"/>
                  <line x1="36" y1="36" x2="24" y2="50" stroke="rgba(0,212,170,0.35)" stroke-width="0.8"/>
                  <line x1="36" y1="36" x2="48" y2="22" stroke="rgba(0,180,216,0.35)" stroke-width="0.8"/>
                  <line x1="36" y1="36" x2="52" y2="34" stroke="rgba(0,180,216,0.3)"  stroke-width="0.8"/>
                  <line x1="36" y1="36" x2="48" y2="50" stroke="rgba(0,180,216,0.35)" stroke-width="0.8"/>
                  <line x1="36" y1="36" x2="36" y2="18" stroke="rgba(0,212,170,0.25)" stroke-width="0.7"/>
                  <line x1="36" y1="36" x2="36" y2="54" stroke="rgba(0,212,170,0.25)" stroke-width="0.7"/>
                  <!-- Curved vessel branches -->
                  <path d="M36 36 Q28 28 22 24" stroke="rgba(0,212,170,0.4)" stroke-width="0.9" fill="none"/>
                  <path d="M36 36 Q44 28 50 24" stroke="rgba(0,180,216,0.4)" stroke-width="0.9" fill="none"/>
                  <path d="M36 36 Q28 44 22 48" stroke="rgba(0,212,170,0.35)" stroke-width="0.9" fill="none"/>
                  <path d="M36 36 Q44 44 50 48" stroke="rgba(0,180,216,0.35)" stroke-width="0.9" fill="none"/>
                  <!-- Optic disc (bright spot) -->
                  <circle cx="43" cy="31" r="4.5" fill="rgba(0,212,170,0.12)" stroke="rgba(0,212,170,0.5)" stroke-width="1"/>
                  <circle cx="43" cy="31" r="2"   fill="rgba(0,212,170,0.3)"/>
                  <!-- Macula (center) -->
                  <circle cx="32" cy="38" r="3.5" fill="none" stroke="rgba(0,180,216,0.4)" stroke-width="0.8" stroke-dasharray="2 2"/>
                  <!-- Pupil / iris -->
                  <circle cx="36" cy="36" r="9"   fill="rgba(0,14,24,0.8)" stroke="#00d4aa" stroke-width="1.5"/>
                  <circle cx="36" cy="36" r="6"   fill="rgba(0,212,170,0.15)"/>
                  <circle cx="36" cy="36" r="3.5" fill="#00d4aa" opacity="0.9"/>
                  <circle cx="36" cy="36" r="1.5" fill="#fff" opacity="0.95"/>
                  <!-- Highlight -->
                  <circle cx="39" cy="33" r="1"   fill="rgba(255,255,255,0.6)"/>
                  <!-- Outer ring glow dots -->
                  <circle cx="36" cy="4"  r="1.2" fill="rgba(0,212,170,0.5)"/>
                  <circle cx="36" cy="68" r="1.2" fill="rgba(0,212,170,0.5)"/>
                  <circle cx="4"  cy="36" r="1.2" fill="rgba(0,212,170,0.4)"/>
                  <circle cx="68" cy="36" r="1.2" fill="rgba(0,212,170,0.4)"/>
                </svg>
            </div>
            <div class="login-logo">NPDR<span style="opacity:0.45">/AI</span></div>
            <div class="login-sub">Retinal Diagnostic Portal &nbsp;&middot;&nbsp; v2.0</div>
        </div>
        """, unsafe_allow_html=True)

        # Tab toggle
        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In",  use_container_width=True,
                         type="primary" if st.session_state.auth_page == "login" else "secondary"):
                st.session_state.auth_page = "login"
                st.rerun()
        with t2:
            if st.button("Register", use_container_width=True,
                         type="primary" if st.session_state.auth_page == "register" else "secondary"):
                st.session_state.auth_page = "register"
                st.rerun()

        st.markdown("<hr style='border-color:#1a2d4a;margin:14px 0;'>", unsafe_allow_html=True)

        # ── LOGIN FORM ──
        if st.session_state.auth_page == "login":
            with st.form("login_form", clear_on_submit=False):
                st.markdown("#### Welcome back")
                username  = st.text_input("Username", placeholder="Enter your username")
                password  = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.warning("Please enter both username and password.")
                    elif do_login(username, password):
                        st.success(f"Welcome back, {st.session_state.current_user}!")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")

            st.markdown("""
            <div style="text-align:center;margin-top:12px;color:#2a3d5a;font-size:0.75rem;">
                Don't have an account? Click <b style="color:#5a8fc0;">Register</b> above.
            </div>
            """, unsafe_allow_html=True)

        # ── REGISTER FORM ──
        else:
            with st.form("register_form", clear_on_submit=True):
                st.markdown("#### Create your account")
                full_name  = st.text_input("Full Name",  placeholder="e.g. Dr. Pratiksha")
                username   = st.text_input("Username",   placeholder="Choose a username (min 3 chars)")
                password   = st.text_input("Password",   type="password", placeholder="Min 6 characters")
                confirm_pw = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                submitted  = st.form_submit_button("Create Account", use_container_width=True)

                if submitted:
                    if password != confirm_pw:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = do_register(full_name, username, password)
                        if ok:
                            st.success(msg)
                            st.info("You can now sign in with your new account.")
                            st.session_state.auth_page = "login"
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("""
            <div style="text-align:center;margin-top:12px;color:#2a3d5a;font-size:0.75rem;">
                Already have an account? Click <b style="color:#5a8fc0;">Sign In</b> above.
            </div>
            """, unsafe_allow_html=True)

    st.stop()

@st.cache_resource
def load_my_model():
    # ── Change this path to your model location ──
    model_path = r"D:\Python_Programs\NPDR\Model_Output_files\final_VGG16_DR_Final.keras"
    if os.path.exists(model_path):
        return load_model(model_path)
    st.error("⚠️  Model file not found. Update `model_path` in the code.")
    return None

model = load_my_model()

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
CLASS_COLORS = {
    "No_DR":           "#3ddc84",
    "Mild":            "#a8e066",
    "Moderate":        "#f5c518",
    "Severe":          "#ff7733",
    "Proliferative_DR":"#ff4444",
}
SEVERITY_CSS = {
    "No_DR":           "severity-no-dr",
    "Mild":            "severity-mild",
    "Moderate":        "severity-moderate",
    "Severe":          "severity-severe",
    "Proliferative_DR":"severity-prolif",
}
SEVERITY_LEVEL = {
    "No_DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferative_DR": 4
}

# ─────────────────────────────────────────────
# MEDICAL DATA
# ─────────────────────────────────────────────
MEDICAL_INFO = {
    "No_DR": {
        "summary": "No visible signs of diabetic retinopathy detected.",
        "findings": ["No microaneurysms observed", "No retinal hemorrhages", "Normal optic disc and vessels", "Clear retinal background"],
        "action": "Routine annual screening recommended.",
        "urgency": "Routine",
        "icd10": "E11.319",
        "follow_up": "12 months",
    },
    "Mild": {
        "summary": "Early-stage non-proliferative diabetic retinopathy.",
        "findings": ["Few microaneurysms (small red dots)", "Minimal retinal damage", "No major hemorrhages", "Localised vascular leakage possible"],
        "action": "Improve glycaemic control. Schedule follow-up in 6-9 months.",
        "urgency": "Low",
        "icd10": "E11.321",
        "follow_up": "6-9 months",
    },
    "Moderate": {
        "summary": "Moderate non-proliferative diabetic retinopathy detected.",
        "findings": ["Increased microaneurysms", "Dot and blot haemorrhages", "Possible hard exudates", "Mild vascular abnormalities"],
        "action": "Ophthalmology referral advised. Optimise HbA1c and blood pressure.",
        "urgency": "Moderate",
        "icd10": "E11.331",
        "follow_up": "3-6 months",
    },
    "Severe": {
        "summary": "Severe non-proliferative diabetic retinopathy - high risk of progression.",
        "findings": ["Extensive haemorrhages in all 4 quadrants", "Venous beading (2+ quadrants)", "Intraretinal microvascular abnormalities (IRMA)", "High risk of conversion to PDR"],
        "action": "Urgent ophthalmology referral. Consider pan-retinal photocoagulation.",
        "urgency": "High",
        "icd10": "E11.341",
        "follow_up": "1-3 months",
    },
    "Proliferative_DR": {
        "summary": "Proliferative diabetic retinopathy - immediate intervention required.",
        "findings": ["Neovascularisation (new abnormal blood vessels)", "Large retinal haemorrhages", "Fibrous tissue / tractional changes", "Imminent risk of vision loss"],
        "action": "URGENT ophthalmology referral. Anti-VEGF therapy or laser photocoagulation.",
        "urgency": "Critical",
        "icd10": "E11.351",
        "follow_up": "Immediately",
    },
}

URGENCY_COLOR = {
    "Routine":  "#3ddc84",
    "Low":      "#a8e066",
    "Moderate": "#f5c518",
    "High":     "#ff7733",
    "Critical": "#ff4444",
}

# ─────────────────────────────────────────────
# HELPERS – preprocessing
# ─────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def assess_image_quality(img: Image.Image) -> dict:
    """Return brightness, contrast, sharpness, blur, and an overall quality score."""
    gray = np.array(img.convert("L"))
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    lap_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # simple blur ratio
    blur_score = min(lap_var / 500.0, 1.0)

    b_score = 1.0 - abs(brightness - 128) / 128.0
    c_score = min(contrast / 60.0, 1.0)
    overall = round((b_score * 0.3 + c_score * 0.3 + blur_score * 0.4) * 100)

    if overall >= 70:
        label = "Good"
    elif overall >= 45:
        label = "Acceptable"
    else:
        label = "Poor"

    return {
        "brightness": round(brightness, 1),
        "contrast":   round(contrast, 1),
        "sharpness":  round(lap_var, 1),
        "blur_score": round(blur_score * 100),
        "overall":    overall,
        "label":      label,
    }


def enhance_image(img: Image.Image, brightness=1.0, contrast=1.0,
                  sharpness=1.0, clahe=False) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if clahe:
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        img = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return img

# ─────────────────────────────────────────────
# HELPERS – Grad-CAM
# ─────────────────────────────────────────────
def make_gradcam_heatmap(img_array: np.ndarray, mdl):
    last_conv = None
    for layer in reversed(mdl.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=mdl.inputs,
        outputs=[mdl.get_layer(last_conv).output, mdl.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.convert_to_tensor(img_array))
        if isinstance(preds, list):
            preds = preds[0]
        idx  = tf.argmax(preds[0])
        loss = preds[:, idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return None

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.maximum(tf.reduce_sum(conv_out[0] * pooled, axis=-1), 0)
    max_v = tf.reduce_max(heatmap)
    if max_v == 0:
        return None
    heatmap = (heatmap / max_v).numpy().astype("float32")
    return heatmap


def overlay_heatmap(orig_bgr: np.ndarray, heatmap,
                    alpha: float = 0.4, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    if heatmap is None or heatmap.ndim != 2 or np.isnan(heatmap).any():
        return orig_bgr
    h, w = orig_bgr.shape[:2]
    hm   = cv2.resize(heatmap.astype("float32"), (w, h))
    hm   = cv2.applyColorMap(np.uint8(255 * hm), colormap)
    return cv2.addWeighted(orig_bgr, 1 - alpha, hm, alpha, 0)

# ─────────────────────────────────────────────
# HELPERS – visualisations
# ─────────────────────────────────────────────
def plot_confidence_bars(probs: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#020d12")
    ax.set_facecolor("#020d12")
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, height=0.45, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", color="#3a6458", fontsize=8)
    ax.tick_params(colors="#7aa89a", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color=(0.0, 0.83, 0.67, 0.06), linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="#e8f4f0", fontsize=7)
    plt.tight_layout(pad=1.0)
    return fig


def plot_severity_gauge(level: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#020d12")
    ax.set_facecolor("#020d12")
    segments  = 5
    seg_colors = ["#00d464", "#a8e066", "#f0c040", "#ff7832", "#ff5064"]
    for i in range(segments):
        theta = np.linspace(np.pi * (1 - i / segments), np.pi * (1 - (i + 1) / segments), 30)
        ax.fill_between(theta, 0.6, 1.0, color=seg_colors[i], alpha=0.85)
    angle = np.pi * (1 - (level + 0.5) / segments)
    ax.plot([angle, angle], [0, 0.58], color="#00d4aa", linewidth=2.5, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("W")
    ax.axis("off")
    ax.set_title("Severity", color="#3a6458", fontsize=9, pad=2)
    plt.tight_layout()
    return fig


def plot_channel_histograms(img: Image.Image) -> plt.Figure:
    arr = np.array(img)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
    fig.patch.set_facecolor("#020d12")
    ch_colors = ["#ff6b8a", "#00d4aa", "#00b4d8"]
    ch_labels = ["Red", "Green", "Blue"]
    for i, (ax, col, lbl) in enumerate(zip(axes, ch_colors, ch_labels)):
        ax.set_facecolor("#020d12")
        ax.hist(arr[:, :, i].ravel(), bins=64, color=col, alpha=0.75)
        ax.set_title(lbl, color=col, fontsize=8)
        ax.spines[:].set_visible(False)
        ax.tick_params(colors="#3a6458", labelsize=6)
    fig.suptitle("RGB Channel Histograms", color="#3a6458", fontsize=9)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# HELPERS – PDF report
# ─────────────────────────────────────────────
def sanitize_pdf_text(text: str) -> str:
    """Replace all non-latin-1 characters with ASCII equivalents for fpdf Helvetica."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00b1": "+/-", # plus-minus
        "\u00b0": " deg",# degree
        "\u2265": ">=",  # greater-equal
        "\u2264": "<=",  # less-equal
        "\u00e9": "e",   # é
        "\u00e8": "e",
        "\u00ea": "e",
        "\u2022": "-",   # bullet
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final fallback: drop anything still outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(patient_name: str, patient_id: str, patient_email: str, result: str,
                         probs: np.ndarray, quality: dict,
                         orig_img: Image.Image, gradcam_arr: np.ndarray) -> bytes:
    info = MEDICAL_INFO[result]
    S = sanitize_pdf_text   # shorthand alias

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(12, 30, 60)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(126, 184, 247)
    pdf.cell(0, 15, "NPDR Detection Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 220)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    # Patient info
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 8, "Patient Information", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 7, S(f"Name: {patient_name or 'N/A'}"))
    pdf.cell(0, 7,  S(f"ID: {patient_id or 'N/A'}"), ln=True)
    pdf.cell(60, 7, S(f"Email: {patient_email or 'N/A'}"))
    pdf.cell(0, 7,  S(f"Date: {datetime.date.today()}"), ln=True)
    pdf.cell(60, 7, S(f"Image Quality: {quality['label']} ({quality['overall']}%)"), ln=True)
    pdf.ln(4)

    # Diagnosis
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 8, "Diagnosis", ln=True, fill=True)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(200, 60, 60) if result != "No_DR" else pdf.set_text_color(30, 150, 80)
    pdf.cell(0, 10, S(f"Classification: {result.replace('_', ' ')}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, S(info["summary"]))
    pdf.ln(4)

    # Confidence table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 8, "Confidence Scores", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    for cls, p in zip(CLASS_NAMES, probs):
        filled = int(p * 30)
        bar = "[" + "|" * filled + "-" * (30 - filled) + "]"
        pdf.cell(60, 6, cls.replace("_", " "))
        pdf.cell(20, 6, f"{p*100:.1f}%")
        pdf.cell(0, 6, bar, ln=True)
    pdf.ln(4)

    # Findings
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 8, "Clinical Findings", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    for finding in info["findings"]:
        pdf.cell(8, 6, "-")
        pdf.cell(0, 6, S(finding), ln=True)
    pdf.ln(4)

    # Recommendations
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 8, "Recommendations", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, S(info["action"]))
    pdf.cell(60, 6, S(f"Urgency: {info['urgency']}"))
    pdf.cell(0, 6,  S(f"Follow-up: {info['follow_up']}"), ln=True)
    pdf.cell(0, 6,  S(f"ICD-10 Code: {info['icd10']}"), ln=True)
    pdf.ln(6)

    # Save images
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf_orig:
        orig_img.resize((200, 200)).save(tf_orig.name, "JPEG")
        orig_path = tf_orig.name

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf_gc:
        Image.fromarray(gradcam_arr).resize((200, 200)).save(tf_gc.name, "JPEG")
        gc_path = tf_gc.name

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 8, "Retinal Images", ln=True, fill=True)
    pdf.image(orig_path, x=10,  y=None, w=85)
    pdf.image(gc_path,  x=110, y=pdf.get_y() - 66, w=85)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(95, 5, "Original Image", align="C")
    pdf.cell(95, 5, "Grad-CAM Heatmap", align="C", ln=True)

    os.unlink(orig_path)
    os.unlink(gc_path)

    # Footer disclaimer
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5,
        "DISCLAIMER: This report is AI-assisted and is intended as a decision-support "
        "tool only. It does not replace professional ophthalmological assessment. "
        "Final diagnosis must be made by a qualified clinician.")

    return bytes(pdf.output())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # User info & logout
    uname = st.session_state.current_user or "User"
    st.markdown(f"""
    <div style="background:rgba(0,212,170,0.05);border:1px solid rgba(0,212,170,0.2);
                border-radius:12px;padding:16px 12px;text-align:center;margin-bottom:10px;
                box-shadow:0 0 20px rgba(0,212,170,0.05);">
        <div style="font-size:1.8rem;margin-bottom:6px;">&#128100;</div>
        <div style="color:#00d4aa;font-family:'DM Mono',monospace;
                    font-size:0.9rem;font-weight:500;">{uname}</div>
        <div style="color:#3a6458;font-size:0.68rem;margin-top:3px;
                    letter-spacing:2px;font-family:'DM Mono',monospace;">
            LOGGED IN
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Sign Out", use_container_width=True):
        do_logout()
        st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:8px 0 8px 0;">
        <svg style="filter:drop-shadow(0 0 8px rgba(0,212,170,0.4));" width="36" height="36" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M6 36 Q20 14 36 14 Q52 14 66 36 Q52 58 36 58 Q20 58 6 36Z"
                fill="rgba(0,212,170,0.06)" stroke="#00d4aa" stroke-width="1.6"/>
          <line x1="36" y1="36" x2="24" y2="22" stroke="rgba(0,212,170,0.4)" stroke-width="1"/>
          <line x1="36" y1="36" x2="48" y2="22" stroke="rgba(0,180,216,0.4)" stroke-width="1"/>
          <line x1="36" y1="36" x2="20" y2="36" stroke="rgba(0,212,170,0.3)" stroke-width="1"/>
          <line x1="36" y1="36" x2="52" y2="36" stroke="rgba(0,180,216,0.3)" stroke-width="1"/>
          <line x1="36" y1="36" x2="24" y2="50" stroke="rgba(0,212,170,0.4)" stroke-width="1"/>
          <line x1="36" y1="36" x2="48" y2="50" stroke="rgba(0,180,216,0.4)" stroke-width="1"/>
          <circle cx="43" cy="31" r="4" fill="rgba(0,212,170,0.15)" stroke="rgba(0,212,170,0.6)" stroke-width="1"/>
          <circle cx="36" cy="36" r="9" fill="rgba(0,10,20,0.9)" stroke="#00d4aa" stroke-width="1.6"/>
          <circle cx="36" cy="36" r="4" fill="#00d4aa" opacity="0.9"/>
          <circle cx="36" cy="36" r="1.8" fill="#fff" opacity="0.95"/>
          <circle cx="39" cy="33" r="1"  fill="rgba(255,255,255,0.5)"/>
        </svg>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(135deg,#00d4aa,#00b4d8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;margin-top:4px;">NPDR<span style="opacity:0.5">/AI</span></div>
        <div style="color:#3a6458;font-size:0.62rem;letter-spacing:3px;
                    text-transform:uppercase;margin-top:2px;font-family:'DM Mono',monospace;">
            v2.0 &middot; VGG16
        </div>
    </div>
    <hr style="border-color:rgba(0,212,170,0.1);margin:10px 0 18px 0;">
    """, unsafe_allow_html=True)

    st.markdown("####  Patient Details")
    st.markdown("<small style='color:#ff6b6b;'>★ All fields are required</small>", unsafe_allow_html=True)
    patient_name  = st.text_input("Patient Name ★", placeholder="Full name")
    patient_id    = st.text_input("Patient ID ★",   placeholder="e.g. PT-2024-001")
    patient_email = st.text_input("Patient Email ★", placeholder="e.g. patient@email.com")
    patient_age   = st.number_input("Age ★", min_value=1, max_value=120, value=None, placeholder="Enter age")
    patient_eye   = st.selectbox("Eye ★", ["— Select —", "Right Eye (OD)", "Left Eye (OS)", "Both"])
    notes         = st.text_area("Clinical Notes", placeholder="Any relevant observations…", height=80)

    st.markdown("---")
    st.markdown("####  Image Enhancement")
    apply_enh   = st.checkbox("Enable Enhancement", value=False)
    brightness  = st.slider("Brightness",  0.5, 2.0, 1.0, 0.05, disabled=not apply_enh)
    contrast    = st.slider("Contrast",    0.5, 2.0, 1.0, 0.05, disabled=not apply_enh)
    sharpness   = st.slider("Sharpness",   0.5, 3.0, 1.0, 0.1,  disabled=not apply_enh)
    use_clahe   = st.checkbox("Apply CLAHE", value=False,        disabled=not apply_enh)

    gc_alpha    = 0.4
    gc_colormap = "JET"
    CM_MAP = {
        "JET": cv2.COLORMAP_JET, "HOT": cv2.COLORMAP_HOT,
        "INFERNO": cv2.COLORMAP_INFERNO, "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
    }

    st.markdown("---")
    # Session stats
    st.markdown("#### Session Stats")
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Scans</h4>
        <span class="value">{st.session_state.total_scans}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        from collections import Counter
        dist = Counter(h["result"] for h in st.session_state.history)
        most_common = dist.most_common(1)[0][0]
        st.markdown(f"""
        <div class="metric-card">
            <h4>Most Frequent</h4>
            <span class="value" style="font-size:1.1rem;">{most_common.replace('_',' ')}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button(" Clear History"):
        st.session_state.history = []
        st.session_state.total_scans = 0
        st.rerun()

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">&#9670; AI-Assisted Diagnostic Tool</div>
    <div class="retinal-logo-wrap">
        <svg class="eye-logo-anim" width="58" height="58" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M6 36 Q20 14 36 14 Q52 14 66 36 Q52 58 36 58 Q20 58 6 36Z"
                fill="rgba(0,212,170,0.06)" stroke="#00d4aa" stroke-width="1.4"/>
          <line x1="36" y1="36" x2="24" y2="22" stroke="rgba(0,212,170,0.35)" stroke-width="0.8"/>
          <line x1="36" y1="36" x2="20" y2="34" stroke="rgba(0,212,170,0.3)"  stroke-width="0.8"/>
          <line x1="36" y1="36" x2="24" y2="50" stroke="rgba(0,212,170,0.35)" stroke-width="0.8"/>
          <line x1="36" y1="36" x2="48" y2="22" stroke="rgba(0,180,216,0.35)" stroke-width="0.8"/>
          <line x1="36" y1="36" x2="52" y2="34" stroke="rgba(0,180,216,0.3)"  stroke-width="0.8"/>
          <line x1="36" y1="36" x2="48" y2="50" stroke="rgba(0,180,216,0.35)" stroke-width="0.8"/>
          <path d="M36 36 Q28 28 22 24" stroke="rgba(0,212,170,0.4)" stroke-width="0.9" fill="none"/>
          <path d="M36 36 Q44 28 50 24" stroke="rgba(0,180,216,0.4)" stroke-width="0.9" fill="none"/>
          <path d="M36 36 Q28 44 22 48" stroke="rgba(0,212,170,0.35)" stroke-width="0.9" fill="none"/>
          <path d="M36 36 Q44 44 50 48" stroke="rgba(0,180,216,0.35)" stroke-width="0.9" fill="none"/>
          <circle cx="43" cy="31" r="4.5" fill="rgba(0,212,170,0.12)" stroke="rgba(0,212,170,0.5)" stroke-width="1"/>
          <circle cx="43" cy="31" r="2"   fill="rgba(0,212,170,0.3)"/>
          <circle cx="32" cy="38" r="3.5" fill="none" stroke="rgba(0,180,216,0.4)" stroke-width="0.8" stroke-dasharray="2 2"/>
          <circle cx="36" cy="36" r="9"   fill="rgba(0,14,24,0.8)" stroke="#00d4aa" stroke-width="1.5"/>
          <circle cx="36" cy="36" r="6"   fill="rgba(0,212,170,0.15)"/>
          <circle cx="36" cy="36" r="3.5" fill="#00d4aa" opacity="0.9"/>
          <circle cx="36" cy="36" r="1.5" fill="#fff" opacity="0.95"/>
          <circle cx="39" cy="33" r="1"   fill="rgba(255,255,255,0.6)"/>
        </svg>
        <div class="logo-text-block">
            <div class="logo-title">NPDR Detection System</div>
            <p style="margin:6px 0 0 0;color:#7aa89a;font-size:0.86rem;font-weight:300;">
                Non-Proliferative Diabetic Retinopathy &nbsp;&middot;&nbsp; VGG16 Deep Learning &nbsp;&middot;&nbsp; Grad-CAM Explainability
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_single, tab_batch, tab_history,tab_comparison, tab_about = st.tabs([
    "🔬 Single Analysis", "📂 Batch Processing", "📋 Session History","📊 Model Comparison", "ℹ️ About"
])

# ══════════════════════════════════════════════
# TAB 1 – SINGLE ANALYSIS
# ══════════════════════════════════════════════
with tab_single:
    upload_col, _ = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Retinal Image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
        )

    if model is None:
        st.error("Model not loaded. Please check the model path in the code.")
        st.stop()

    if uploaded_file:
        # ── Validate required patient fields ──
        import re
        missing = []
        if not patient_name or not patient_name.strip():
            missing.append("Patient Name")
        if not patient_id or not patient_id.strip():
            missing.append("Patient ID")
        if not patient_email or not patient_email.strip():
            missing.append("Patient Email")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", patient_email.strip()):
            st.error("⚠️ Please enter a valid email address for the patient.")
            st.stop()
        if patient_age is None:
            missing.append("Age")
        if patient_eye == "— Select —":
            missing.append("Eye")
        if missing:
            st.error(f"⚠️ Please fill in all required patient details before analysis: **{', '.join(missing)}**")
            st.stop()

        # Use file name as a key so history only appends once per unique upload
        upload_key = uploaded_file.name + str(uploaded_file.size)

        st.markdown("---")

        # ── Step 1: Open image ──
        step1 = st.info("📂 Step 1/5 — Opening image…")
        try:
            img_raw = Image.open(uploaded_file).convert("RGB")
            step1.success(f"✅ Step 1/5 — Image opened: {img_raw.size[0]}×{img_raw.size[1]} px")
        except Exception as e:
            step1.error(f"❌ Step 1 failed — Could not open image: {e}")
            st.stop()

        # ── Step 2: Enhancement ──
        step2 = st.info("🎨 Step 2/5 — Applying enhancement…")
        try:
            img_display = enhance_image(img_raw, brightness, contrast, sharpness, use_clahe) \
                          if apply_enh else img_raw
            step2.success("✅ Step 2/5 — Enhancement done")
        except Exception as e:
            step2.warning(f"⚠️ Step 2 — Enhancement failed, using original: {e}")
            img_display = img_raw

        # ── Step 3: Quality assessment ──
        step3 = st.info("🔎 Step 3/5 — Assessing image quality…")
        try:
            quality = assess_image_quality(img_display)
            step3.success(f"✅ Step 3/5 — Quality: {quality['label']} ({quality['overall']}%)")
        except Exception as e:
            step3.warning(f"⚠️ Step 3 — Quality check failed: {e}")
            quality = {"brightness": 0, "contrast": 0, "sharpness": 0,
                       "blur_score": 0, "overall": 0, "label": "Unknown"}
        q_color_map = {"Good": "#3ddc84", "Acceptable": "#f5c518",
                       "Poor": "#ff4444", "Unknown": "#667799"}
        q_color = q_color_map.get(quality["label"], "#667799")

        # ── Step 4: Preprocess & predict ──
        step4 = st.info("🧠 Step 4/5 — Running AI model…")
        try:
            img_array  = preprocess_image(img_display)
            prediction = model.predict(img_array, verbose=0)
            if isinstance(prediction, list):
                prediction = prediction[0]
            probs           = prediction[0]
            predicted_class = int(np.argmax(probs))
            result          = CLASS_NAMES[predicted_class]
            confidence      = float(probs[predicted_class])
            info            = MEDICAL_INFO[result]
            step4.success(f"✅ Step 4/5 — Prediction: **{result}** ({confidence*100:.1f}% confidence)")
        except Exception as e:
            step4.error(f"❌ Step 4 FAILED — Model prediction error: {e}")
            st.exception(e)
            st.stop()

        # ── Step 5: Grad-CAM ──
        step5 = st.info("🌡️ Step 5/5 — Generating Grad-CAM heatmap…")
        try:
            heatmap  = make_gradcam_heatmap(img_array, model)
            orig_bgr = cv2.cvtColor(np.array(img_display), cv2.COLOR_RGB2BGR)
            gc_bgr   = overlay_heatmap(orig_bgr, heatmap, gc_alpha, CM_MAP[gc_colormap])
            gc_rgb   = cv2.cvtColor(gc_bgr, cv2.COLOR_BGR2RGB)
            gc_pil   = Image.fromarray(gc_rgb)
            step5.success("✅ Step 5/5 — Grad-CAM ready")
        except Exception as e:
            step5.warning(f"⚠️ Step 5 — Grad-CAM failed, showing original: {e}")
            gc_rgb = np.array(img_display)
            gc_pil = img_display

        # Clear step indicators and show full results
        step1.empty(); step2.empty(); step3.empty(); step4.empty(); step5.empty()

        # ── Step 6: Save to history (only once per upload) ──
        if upload_key not in [h.get("key") for h in st.session_state.history]:
            st.session_state.total_scans += 1
            st.session_state.history.append({
                "key":        upload_key,
                "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
                "patient":    patient_name or "Anonymous",
                "result":     result,
                "confidence": confidence,
                "eye":        patient_eye,
                "scanned_by": st.session_state.current_user or "Unknown",
            })

        # ── Severity badge ──
        sev_css = SEVERITY_CSS[result]
        urgency_col = URGENCY_COLOR[info["urgency"]]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;margin:8px 0 18px 0;flex-wrap:wrap;">
            <span class="severity-badge {sev_css}">{result.replace('_', ' ')}</span>
            <span style="color:{q_color};font-size:0.85rem;">
                ● Image Quality: {quality['label']} ({quality['overall']}%)
            </span>
            <span style="color:{urgency_col};font-size:0.85rem;">
                ⚑ Urgency: {info['urgency']}
            </span>
            <span style="color:#667799;font-size:0.85rem;">
                ICD-10: {info['icd10']}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Image columns ──
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">ORIGINAL</div>', unsafe_allow_html=True)
            st.image(img_display, use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">GRAD-CAM HEATMAP</div>', unsafe_allow_html=True)
            st.image(gc_pil, use_container_width=True)
        with c3:
            st.markdown('<div class="section-title">IMAGE ANALYSIS</div>', unsafe_allow_html=True)
            fig_hist = plot_channel_histograms(img_display)
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
            st.markdown(f"""
            <div class="metric-card" style="margin-top:8px;">
                <h4>Quality Metrics</h4>
                Brightness <b>{quality['brightness']}</b> &nbsp;|&nbsp;
                Contrast <b>{quality['contrast']}</b><br>
                Sharpness <b>{quality['sharpness']}</b> &nbsp;|&nbsp;
                Blur Score <b>{quality['blur_score']}%</b>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence & gauge ──
        st.markdown('<div class="section-title">CONFIDENCE SCORES</div>', unsafe_allow_html=True)
        ga, gb = st.columns([3, 1])
        with ga:
            fig_conf = plot_confidence_bars(probs)
            st.pyplot(fig_conf, use_container_width=True)
            plt.close(fig_conf)
        with gb:
            fig_gauge = plot_severity_gauge(SEVERITY_LEVEL[result])
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <h4>Confidence</h4>
                <span class="value" style="color:{CLASS_COLORS[result]};">{confidence*100:.1f}</span>
                <span class="unit">%</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Medical interpretation ──
        st.markdown('<div class="section-title">MEDICAL INTERPRETATION</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            findings_html = "".join(f"<li>{f}</li>" for f in info["findings"])
            st.markdown(f"""
            <div class="med-box">
                <div class="finding">🔍 Clinical Findings</div>
                <ul style="margin:8px 0 0 16px;padding:0;">{findings_html}</ul>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="med-box">
                <div class="finding">📋 Summary</div>
                <p style="margin:6px 0;">{info['summary']}</p>
                <div class="finding" style="margin-top:10px;">✅ Recommended Action</div>
                <p style="margin:6px 0;">{info['action']}</p>
                <div class="finding" style="margin-top:10px;">📅 Follow-up</div>
                <p style="margin:6px 0;">{info['follow_up']}</p>
            </div>
            """, unsafe_allow_html=True)

        if notes:
            st.markdown(f"""
            <div class="med-box" style="margin-top:10px;">
                <div class="finding">📝 Clinical Notes</div>
                <p style="margin:6px 0;">{notes}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Download PDF report ──
        st.markdown('<div class="section-title">REPORT</div>', unsafe_allow_html=True)
        if st.button("📄 Generate & Download PDF Report"):
            with st.spinner("Generating report…"):
                pdf_bytes = generate_pdf_report(
                    patient_name, patient_id, patient_email, result, probs, quality,
                    img_display, gc_rgb
                )
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"NPDR_Report_{patient_id or 'unknown'}_{datetime.date.today()}.pdf",
                mime="application/pdf",
            )

        # ── JSON export ──
        export_data = {
            "patient":       patient_name,
            "patient_id":    patient_id,
            "patient_email": patient_email,
            "age":           int(patient_age),
            "eye":           patient_eye,
            "timestamp":  datetime.datetime.now().isoformat(),
            "result":     result,
            "confidence": round(float(confidence), 4),
            "all_scores": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
            "image_quality": quality,
            "icd10":      info["icd10"],
            "follow_up":  info["follow_up"],
            "urgency":    info["urgency"],
            "notes":      notes,
        }
        st.download_button(
            label="⬇️ Export JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"NPDR_{patient_id or 'unknown'}_{datetime.date.today()}.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════
# TAB 2 – BATCH PROCESSING
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📂 Batch Retinal Image Analysis")
    st.info("Upload multiple retinal images for simultaneous analysis.")

    batch_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if batch_files and model is not None:
        if st.button("▶️ Run Batch Analysis"):
            results_list = []
            prog = st.progress(0, text="Analysing…")

            for i, bf in enumerate(batch_files):
                bimg = Image.open(bf).convert("RGB")
                barr = preprocess_image(bimg)
                bpred = model.predict(barr, verbose=0)
                if isinstance(bpred, list):
                    bpred = bpred[0]
                bprobs   = bpred[0]
                bcls     = int(np.argmax(bprobs))
                bresult  = CLASS_NAMES[bcls]
                bconf    = float(bprobs[bcls])
                bquality = assess_image_quality(bimg)

                results_list.append({
                    "File":       bf.name,
                    "Result":     bresult,
                    "Confidence": f"{bconf*100:.1f}%",
                    "Quality":    bquality["label"],
                    "Urgency":    MEDICAL_INFO[bresult]["urgency"],
                })
                prog.progress((i + 1) / len(batch_files), text=f"Processed {i+1}/{len(batch_files)}")

            prog.empty()
            st.success(f"✅ Analysed {len(results_list)} images")

            # Summary stats
            from collections import Counter
            dist = Counter(r["Result"] for r in results_list)
            cols = st.columns(len(CLASS_NAMES))
            for col, cls in zip(cols, CLASS_NAMES):
                col.metric(cls.replace("_", " "), dist.get(cls, 0))

            # Results table
            st.markdown("#### Results")
            import pandas as pd
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)

            # CSV download
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                f"NPDR_Batch_{datetime.date.today()}.csv",
                "text/csv",
            )

# ══════════════════════════════════════════════
# TAB 3 – SESSION HISTORY
# ══════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 Session Scan History")

    if not st.session_state.history:
        st.info("No scans in this session yet. Analyse an image in the Single Analysis tab.")
    else:
        hist = st.session_state.history

        # Summary cards
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Scans", len(hist))
        h2.metric("DR Positive", sum(1 for h in hist if h["result"] != "No_DR"))
        h3.metric("Avg Confidence", f"{np.mean([h['confidence'] for h in hist])*100:.1f}%")
        h4.metric("Critical Cases", sum(1 for h in hist if h["result"] == "Proliferative_DR"))

        # History list
        st.markdown('<div class="section-title">SCAN LOG</div>', unsafe_allow_html=True)
        for i, h in enumerate(reversed(hist)):
            col = CLASS_COLORS[h["result"]]
            sev = SEVERITY_CSS[h["result"]]
            st.markdown(f"""
            <div class="hist-row">
                <span style="color:#44617a;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;">
                    #{len(hist)-i:03d}
                </span>
                <span style="color:#8899bb;">🕐 {h.get('timestamp','--')}</span>
                <span style="flex:1;color:#dde3f0;">{h.get('patient','—')}</span>
                <span class="severity-badge {sev}" style="padding:3px 10px;font-size:0.76rem;">
                    {h['result'].replace('_',' ')}
                </span>
                <span style="color:{col};font-size:0.83rem;">{h['confidence']*100:.1f}%</span>
                <span style="color:#5a8fc0;font-size:0.78rem;">{h.get('scanned_by','')}</span>
                <span style="color:#44617a;font-size:0.8rem;">{h.get('eye','')}</span>
            </div>
            """, unsafe_allow_html=True)

        # Distribution chart
        st.markdown('<div class="section-title">DISTRIBUTION</div>', unsafe_allow_html=True)
        from collections import Counter
        dist  = Counter(h["result"] for h in hist)
        vals  = [dist.get(c, 0) for c in CLASS_NAMES]
        fig2, ax2 = plt.subplots(figsize=(7, 2.8))
        fig2.patch.set_facecolor("#020d12")
        ax2.set_facecolor("#020d12")
        bars2 = ax2.bar(CLASS_NAMES, vals,
                        color=[CLASS_COLORS[c] for c in CLASS_NAMES], width=0.5)
        ax2.set_ylabel("Count", color="#3a6458", fontsize=8)
        ax2.tick_params(colors="#7aa89a", labelsize=8)
        ax2.spines[:].set_visible(False)
        ax2.yaxis.grid(True, color=(0.0, 0.83, 0.67, 0.06), linewidth=0.5)
        for bar, v in zip(bars2, vals):
            if v:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         str(v), ha="center", color="#e8f4f0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
# ══════════════════════════════════════════════
# TAB 4 – MODEL COMPARISON  ← NEW TAB
# ══════════════════════════════════════════════
with tab_comparison:
    st.markdown("### 📊 Model Performance Comparison")
    st.markdown("Benchmark accuracy across three architectures trained on the DR grading dataset.")
 
    # ── Metric cards ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card" style="text-align:center;">
            <h4>VGG16 (Deployed)</h4>
            <span class="value" style="color:#00d4aa;">87</span><span class="unit">%</span>
            <p style="color:#3ddc84;font-size:0.75rem;margin-top:8px;margin-bottom:0;">★ Best accuracy</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card" style="text-align:center;">
            <h4>EfficientNet-B0</h4>
            <span class="value" style="color:#f0c060;">74</span><span class="unit">%</span>
            <p style="color:#7aa89a;font-size:0.75rem;margin-top:8px;margin-bottom:0;">−13% vs VGG16</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card" style="text-align:center;">
            <h4>Custom CNN</h4>
            <span class="value" style="color:#ff7832;">69</span><span class="unit">%</span>
            <p style="color:#7aa89a;font-size:0.75rem;margin-top:8px;margin-bottom:0;">−18% vs VGG16</p>
        </div>""", unsafe_allow_html=True)
 
    # ── Bar chart ──
    st.markdown('<div class="section-title">TEST ACCURACY COMPARISON</div>', unsafe_allow_html=True)
 
    models_list  = ["VGG16", "EfficientNet-B0", "Custom CNN"]
    accuracies   = [87, 74, 69]
    bar_colors   = ["#00d4aa", "#f0c060", "#ff7832"]
 
    fig_bar, ax_bar = plt.subplots(figsize=(7, 3.8))
    fig_bar.patch.set_facecolor("#020d12")
    ax_bar.set_facecolor("#020d12")
    bars = ax_bar.bar(models_list, accuracies, color=bar_colors, width=0.45, zorder=3,
                      edgecolor=["#008f72", "#c09a00", "#cc5500"], linewidth=0.8)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Accuracy (%)", color="#3a6458", fontsize=9)
    ax_bar.tick_params(colors="#7aa89a", labelsize=10)
    ax_bar.spines[:].set_visible(False)
    ax_bar.yaxis.grid(True, color=(0.0, 0.83, 0.67, 0.06), linewidth=0.5, zorder=0)
    ax_bar.set_axisbelow(True)
    # Accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc}%", ha="center", va="bottom", color="#e8f4f0",
                    fontsize=13, fontweight="bold")

    # Baseline reference line
    ax_bar.axhline(y=87, color="#00d4aa", linewidth=0.8, linestyle="--", alpha=0.4, zorder=2)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)
    
# ── Accuracy gap visual ──
    st.markdown('<div class="section-title">ACCURACY GAP VS VGG16</div>', unsafe_allow_html=True)
 
    gap_col1, gap_col2 = st.columns(2)
    with gap_col1:
        st.markdown("""
        <div class="med-box">
            <div class="finding">EfficientNet-B0 gap</div>
            <p style="margin:8px 0 4px 0;font-size:1.6rem;color:#f0c060;font-weight:700;">−13%</p>
            <div style="height:10px;background:rgba(255,255,255,0.05);border-radius:5px;margin-top:6px;">
                <div style="height:100%;width:13%;background:#f0c060;border-radius:5px;"></div>
            </div>
            <p style="margin:6px 0 0 0;font-size:0.8rem;color:#5a7a6a;">
                Lightweight model — 5.3M params vs 138M. Good for edge deployment but trades accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with gap_col2:
        st.markdown("""
        <div class="med-box">
            <div class="finding">Custom CNN gap</div>
            <p style="margin:8px 0 4px 0;font-size:1.6rem;color:#ff7832;font-weight:700;">−18%</p>
            <div style="height:10px;background:rgba(255,255,255,0.05);border-radius:5px;margin-top:6px;">
                <div style="height:100%;width:18%;background:#ff7832;border-radius:5px;"></div>
            </div>
            <p style="margin:6px 0 0 0;font-size:0.8rem;color:#5a7a6a;">
                Trained from scratch without ImageNet pretraining — limited by dataset size.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary table ──
    st.markdown('<div class="section-title">SUMMARY TABLE</div>', unsafe_allow_html=True)
    import pandas as pd
    df_compare = pd.DataFrame({
        "Model":         ["VGG16 ✅", "EfficientNet-B0", "Custom CNN"],
        "Accuracy (%)":  [87, 74, 69],
        "Parameters":    ["138M", "5.3M", "~2M"],
        "Input Size":    ["224×224", "224×224", "224×224"],
        "Pretrained":    ["ImageNet", "ImageNet", "No"],
        "Augmentation":  ["Yes", "Yes", "Yes"],
        "Notes":         [
            "Transfer learning, fine-tuned — deployed model",
            "Lightweight, mobile-friendly — lower accuracy",
            "Baseline trained from scratch — limited generalisation",
        ],
    })
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
 
    st.markdown("""
    <div class="med-box" style="margin-top:16px;">
        <div class="finding">📌 Conclusion</div>
        <p style="margin:8px 0 0 0;">
            VGG16 with ImageNet transfer learning achieves the highest test accuracy of <b style="color:#00d4aa;">87%</b>
            on the DR grading task. While EfficientNet-B0 offers a substantially smaller parameter footprint
            (5.3M vs 138M), the 13% accuracy gap makes it unsuitable for clinical deployment without further
            fine-tuning. The custom CNN baseline confirms that pretraining on large datasets is critical for
            retinal image classification with limited medical data.
        </p>
    </div>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════
# TAB 4 – ABOUT
# ══════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ### 🩺 About This System

    **NPDR Detection System** is an AI-powered clinical decision-support tool for the
    automated grading of *Non-Proliferative Diabetic Retinopathy* from fundus photographs.

    #### 🧠 Model Architecture
    - **Base**: VGG16 (ImageNet pre-trained)
    - **Input**: 224 × 224 RGB fundus images
    - **Output**: 5-class softmax (No DR → Proliferative DR)
    - **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM)

    #### 📊 Classification Scale
    | Grade | Class | Description |
    |-------|-------|-------------|
    | 0 | No DR | No visible retinopathy |
    | 1 | Mild NPDR | Microaneurysms only |
    | 2 | Moderate NPDR | More than mild, less than severe |
    | 3 | Severe NPDR | Extensive haemorrhages / IRMA |
    | 4 | Proliferative DR | Neovascularisation present |

    #### ✨ Features
    - Real-time single-image classification with Grad-CAM overlay
    - Image quality assessment (brightness, contrast, sharpness)
    - Interactive image enhancement (CLAHE, brightness, contrast)
    - Customisable Grad-CAM colormap and opacity
    - Detailed medical interpretation per grade
    - PDF report generation with patient metadata
    - JSON data export
    - Batch processing with CSV export
    - Session history and analytics dashboard

    #### ⚠️ Disclaimer
    > This tool is **not** a substitute for professional medical advice. All results must
    > be reviewed and confirmed by a qualified ophthalmologist before clinical action is taken.

    ---
    Built with Streamlit · TensorFlow/Keras · OpenCV · FPDF · Matplotlib
    """)
