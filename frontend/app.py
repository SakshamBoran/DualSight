"""DualSight web interface (Streamlit).

Run from the project folder, with the backend already running:
    streamlit run frontend/app.py
"""
import base64
import io
import os
import random

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_BASE_URL = os.environ.get("AEROPRICE_API", "http://localhost:5000")

# ------------------------------------------------------------------ theme
st.set_page_config(page_title="DualSight", layout="wide", initial_sidebar_state="collapsed")
DARK = bool(st.session_state.get("dark_mode", False))

# Pastel tints borrowed from map land-cover legends, each with a deep partner for text and marks.
# The dark palette keeps the same hues as deep tints with light partners.
LIGHT = dict(
    INK="#1F2A37", MUTED="#5D6B7A", RULE="#D9DEEA", BG="#F7F8FC", WHITE="#FFFFFF", EDGE="#B7C1D3",
    MINT="#D7F2E3", MINT_D="#2E8A5E", SKY="#DCEBFA", SKY_D="#2F6EA5", PEACH="#FFE4D4", PEACH_D="#C2603A",
    BUTTER="#FFF2C2", BUTTER_D="#8F6B0B", LAVENDER="#E8E1FA", LAVENDER_D="#6A4FB0", ROSE="#FBDDE4", ROSE_D="#B83A57",
    UP_BAR="#7CCFA3", DOWN_BAR="#F29BAE", BASE_BAR="#8DB8E6", UP_STRONG="#3FAE77", DOWN_STRONG="#E2577A",
    BTN="#2F6EA5", BTN_EDGE="#245A8A", BTN_SHADOW="#1B4570", STAR="#F2C14E", LOW_STEP="#F3EEF1",
    EXTRA1="#D8F0F2", EXTRA2="#F4E3F0", DOT_ALPHA=.3,
)
DARK_P = dict(
    INK="#E6EBF4", MUTED="#9AA6B8", RULE="#2F3A50", BG="#111827", WHITE="#1A2334", EDGE="#43506A",
    MINT="#1D3A2E", MINT_D="#86DDB0", SKY="#1C2F48", SKY_D="#93C2F0", PEACH="#3D2A22", PEACH_D="#F4B492",
    BUTTER="#3A3219", BUTTER_D="#F1D27E", LAVENDER="#2B2645", LAVENDER_D="#C1AEF4", ROSE="#3E2130", ROSE_D="#F5A0B5",
    UP_BAR="#5DBE8C", DOWN_BAR="#E07F97", BASE_BAR="#6E9FD6", UP_STRONG="#7FE0AE", DOWN_STRONG="#FF8FA9",
    BTN="#3E7FC0", BTN_EDGE="#5A97D2", BTN_SHADOW="#1F3F63", STAR="#F2C14E", LOW_STEP="#2A2433",
    EXTRA1="#173338", EXTRA2="#3A2236", DOT_ALPHA=.45,
)
_P = DARK_P if DARK else LIGHT
(INK, MUTED, RULE, BG, WHITE, EDGE, MINT, MINT_D, SKY, SKY_D, PEACH, PEACH_D, BUTTER, BUTTER_D, LAVENDER, LAVENDER_D,
 ROSE, ROSE_D, UP_BAR, DOWN_BAR, BASE_BAR, UP_STRONG, DOWN_STRONG, BTN, BTN_EDGE, BTN_SHADOW, STAR, LOW_STEP,
 EXTRA1, EXTRA2, DOT_ALPHA) = (_P[k] for k in (
    "INK", "MUTED", "RULE", "BG", "WHITE", "EDGE", "MINT", "MINT_D", "SKY", "SKY_D", "PEACH", "PEACH_D", "BUTTER",
    "BUTTER_D", "LAVENDER", "LAVENDER_D", "ROSE", "ROSE_D", "UP_BAR", "DOWN_BAR", "BASE_BAR", "UP_STRONG", "DOWN_STRONG",
    "BTN", "BTN_EDGE", "BTN_SHADOW", "STAR", "LOW_STEP", "EXTRA1", "EXTRA2", "DOT_ALPHA"))
FONT = "Archivo, Helvetica Neue, Arial, sans-serif"


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# Small line icons (drawn in the current text colour)
_ICON_PATHS = {
    "area": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9h3M4 14h3M9 4v3M14 4v3"/>',
    "bed": '<path d="M3 18v-8M3 14h18v4M21 18v-4a3 3 0 0 0-3-3h-7v3"/><circle cx="7" cy="11.5" r="1.5"/>',
    "bath": '<path d="M4 12h16v3a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4z"/><path d="M6 12V6a2 2 0 0 1 4 0"/><path d="M7 19l-1 2M17 19l1 2"/>',
    "calendar": '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M4 10h16M9 3v4M15 3v4"/>',
    "tree": '<path d="M12 21v-6"/><path d="M12 3a6 6 0 0 1 5 9.5A4 4 0 0 1 13 17h-2a4 4 0 0 1-4-4.5A6 6 0 0 1 12 3z"/>',
    "school": '<path d="M3 9l9-5 9 5-9 5z"/><path d="M7 11.5V16c0 1.5 2.5 3 5 3s5-1.5 5-3v-4.5"/>',
    "health": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M12 8v8M8 12h8"/>',
    "city": '<path d="M3 21h18M5 21V10h5v11M10 21V4h9v17M13 8h3M13 12h3M13 16h3M7 13h1M7 16h1"/>',
    "bus": '<rect x="5" y="3" width="14" height="14" rx="3"/><path d="M5 11h14M8 21l1-4M16 21l-1-4"/>'
           '<circle cx="9" cy="14" r=".8"/><circle cx="15" cy="14" r=".8"/>',
    "pin": '<path d="M12 21s-6-5.5-6-11a6 6 0 0 1 12 0c0 5.5-6 11-6 11z"/><circle cx="12" cy="10" r="2"/>',
}


def icon(name, size=16):
    return (f'<svg class="ds-ic" viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>')


st.markdown(f"""
<style>
@font-face {{
    font-family: "Archivo";
    src: url("app/static/fonts/archivo-variable.woff2") format("woff2");
    font-weight: 100 900; font-stretch: 62% 125%; font-display: swap;
}}
html, body, [data-testid="stAppViewContainer"], button, input, textarea, select {{ font-family: {FONT}; }}
[data-testid="stAppDeployButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebar"] {{ display: none; }}
.block-container {{ padding-top: 2.6rem; max-width: 1260px; }}

/* Page and Streamlit widgets follow the chosen theme */
.stApp, [data-testid="stAppViewContainer"] {{ background: {BG}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; pointer-events: none; }}
[data-testid="stToolbar"] {{ pointer-events: auto; }}
[data-testid="stToolbar"] svg {{ fill: {MUTED}; color: {MUTED}; }}
[data-testid="stMarkdownContainer"] h5, [data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] p {{ color: {INK}; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{ color: {INK} !important; }}
[data-testid="stSelectbox"] [role="group"], [data-baseweb="select"] > div {{
    background-color: {WHITE} !important; border: 1.5px solid {EDGE} !important; border-radius: 12px !important; }}
[data-testid="stSelectbox"] [role="group"]:focus-within {{ border-color: {SKY_D} !important; }}
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] [role="group"] div, [data-baseweb="select"] div {{ color: {INK} !important; }}
[data-testid="stSelectbox"] svg {{ fill: {INK}; color: {INK}; }}
[role="listbox"], [role="listbox"] [role="option"], [data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{
    background: {WHITE} !important; color: {INK} !important; }}
[role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] li:hover {{ background: {SKY} !important; }}
[data-testid="stTabs"] [role="tab"], [data-testid="stTabs"] [role="tab"] p {{ color: {MUTED} !important; }}
[data-testid="stTabs"] [role="tab"]:hover p {{ color: {INK} !important; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"], [data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: {SKY_D} !important; }}
[data-baseweb="tab-highlight"] {{ background-color: {SKY_D} !important; }}
[data-baseweb="tab-border"] {{ background-color: {RULE} !important; }}
[data-testid="stAlertContainer"], [data-testid="stAlert"] > div {{ background-color: {SKY} !important; color: {INK} !important; }}
[data-testid="stAlertContainer"] p {{ color: {INK} !important; }}
[data-testid="stSpinner"], [data-testid="stSpinner"] p {{ color: {INK}; }}

/* Header */
.ds-head {{ text-align: center; padding: .2rem 0 1.4rem; margin-bottom: 1.5rem; border-bottom: 1px solid {RULE}; }}
.ds-mark {{ display: block; margin: 0 auto .35rem; }}
.ds-word {{ font-stretch: 125%; font-weight: 800; font-size: clamp(2.5rem, 6.5vw, 3.8rem); letter-spacing: -.015em;
            line-height: 1.08; color: {INK}; }}
.ds-tag {{ max-width: 58ch; margin: .55rem auto 1rem; color: {MUTED}; font-size: 1.06rem; line-height: 1.5; }}
.ds-flow {{ display: inline-flex; flex-wrap: wrap; gap: .45rem; justify-content: center; align-items: center; font-size: .9rem; }}
.ds-flow .pill {{ border-radius: 999px; padding: 5px 13px; font-weight: 600; }}
.ds-flow .op {{ color: {MUTED}; font-weight: 700; }}
.ds-ic {{ vertical-align: -3px; margin-right: 6px; }}

.ap-empty {{ background: {WHITE}; border: 2px dashed {EDGE}; border-radius: 18px; padding: 2.2rem; text-align: center;
             color: {MUTED}; font-size: 1.05rem; margin-top: 1rem; }}
.ap-empty b {{ color: {INK}; }}

.ap-plate {{ background: {WHITE}; border: 1px solid {RULE}; border-radius: 18px; padding: 12px 12px 9px; }}
.ap-plate img {{ width: 100% !important; max-width: 100% !important; height: auto !important; aspect-ratio: 1 / 1;
                 object-fit: cover; display: block; border-radius: 10px; }}
.ap-plate-caption {{ display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; color: {MUTED};
                     font-size: .8rem; padding: 8px 4px 0; font-variant-numeric: tabular-nums; }}

.ap-address {{ font-stretch: 112%; font-weight: 650; font-size: 1.45rem; color: {INK}; line-height: 1.2; margin: 0; }}
.ap-zip {{ color: {MUTED}; font-size: .95rem; margin: .15rem 0 .9rem; }}
.ap-estimate {{ background: {SKY}; border-radius: 18px; padding: 16px 20px 12px; margin-bottom: 1rem; }}
.ap-label {{ color: {SKY_D}; font-size: .9rem; font-weight: 600; margin: 0; }}
.ap-price {{ font-stretch: 125%; font-weight: 700; font-size: clamp(2.3rem, 5vw, 3.4rem); color: {INK}; line-height: 1.02;
             font-variant-numeric: tabular-nums lining-nums; letter-spacing: -.015em; margin: .15rem 0 .25rem; }}
.ap-listed {{ font-size: 1.02rem; color: {INK}; font-variant-numeric: tabular-nums; margin: 0 0 .6rem; }}

.ap-scale {{ position: relative; height: 62px; margin: .2rem 0 .3rem; }}
.ap-scale-track {{ position: absolute; left: 0; right: 0; top: 26px; height: 8px; border: 1.5px solid {INK};
                   background: repeating-linear-gradient(90deg, {INK} 0 10%, {WHITE} 10% 20%); }}
.ap-scale-gap {{ position: absolute; top: 24px; height: 12px; border: 1.5px solid {INK}; }}
.ap-tick {{ position: absolute; transform: translateX(-50%); font-size: .78rem; white-space: nowrap; color: {INK}; }}
.ap-tick.est {{ top: 0; }}
.ap-tick.est::after {{ content: ""; display: block; margin: 2px auto 0; width: 0; height: 0; border-left: 6px solid transparent;
                       border-right: 6px solid transparent; border-top: 8px solid {INK}; }}
.ap-tick.lst {{ top: 38px; }}
.ap-tick.lst::before {{ content: ""; display: block; margin: 0 auto 2px; width: 0; height: 0; border-left: 6px solid transparent;
                        border-right: 6px solid transparent; border-bottom: 8px solid {MUTED}; }}
.ap-scale-ends {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .72rem; font-variant-numeric: tabular-nums; }}

.ap-chip {{ display: inline-block; border-radius: 999px; padding: 4px 12px; font-size: .85rem; font-weight: 600; margin: .45rem .3rem .2rem 0; }}

.ap-tiles {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .6rem; margin: .2rem 0 1rem; }}
.ap-tile {{ border-radius: 14px; padding: 10px 12px; }}
.ap-tile span {{ display: block; font-size: .8rem; font-weight: 600; }}
.ap-tile b {{ font-weight: 650; font-size: 1.05rem; color: {INK}; font-variant-numeric: tabular-nums; }}

.ap-section {{ font-stretch: 112%; font-weight: 700; font-size: 1.55rem; color: {INK}; margin: 2.2rem 0 .3rem; }}
.ap-lead {{ font-size: 1.04rem; line-height: 1.55; max-width: 74ch; color: {INK}; }}
.ap-note {{ color: {MUTED}; font-size: .86rem; line-height: 1.5; max-width: 80ch; }}
.ap-howto {{ background: {BUTTER}; border-radius: 14px; padding: 11px 15px; font-size: .9rem; line-height: 1.5; color: {INK};
             margin: .2rem 0 1rem; }}
.ap-howto b {{ color: {BUTTER_D}; }}
.ap-card {{ background: {WHITE}; border: 1px solid {RULE}; border-radius: 16px; padding: 14px 16px; margin-bottom: .7rem; color: {INK}; }}

.ap-factors {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: .7rem; margin-top: .4rem; }}
.ap-factor {{ border-radius: 16px; padding: 12px 15px; line-height: 1.45; color: {INK}; }}
.ap-factor .ttl {{ margin: 0 0 .15rem; font-size: 1rem; color: {INK}; }}
.ap-factor .amt {{ font-stretch: 112%; font-weight: 700; font-size: 1.35rem; font-variant-numeric: tabular-nums; }}
.ap-factor p {{ margin: .25rem 0 0; font-size: .88rem; color: {INK}; }}

.ap-squares li {{ margin: .25rem 0; font-variant-numeric: tabular-nums; }}
.ap-tag {{ display: inline-block; min-width: 2.1em; text-align: center; border-radius: 7px; font-weight: 700; font-size: .8rem;
           padding: 1px 6px; margin-right: .4rem; }}
.ap-swatch {{ display: inline-block; width: .85em; height: .85em; border-radius: 3px; margin-right: .45rem; vertical-align: -1px; }}

.ap-amenities {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .7rem; }}
.ap-amenity {{ border-radius: 16px; padding: 14px 16px; color: {INK}; }}
.ap-amenity .badge {{ display: inline-flex; align-items: center; justify-content: center; width: 42px; height: 42px;
                      border-radius: 50%; background: {WHITE}; }}
.ap-amenity .badge .ds-ic {{ margin: 0; vertical-align: 0; }}
.ap-amenity .ttl {{ margin: .45rem 0 .1rem; font-size: 1rem; color: {INK}; }}
.ap-amenity p {{ margin: .2rem 0; font-size: .88rem; color: {INK}; }}
.ap-dots span {{ display: inline-block; width: 22px; height: 7px; border-radius: 9px; margin-right: 4px; }}

/* Buttons: solid, slightly raised, with a firm border */
.stButton > button {{ font-family: {FONT}; border-radius: 12px; min-height: 2.85rem; padding: .55rem 1.2rem;
                      background: {WHITE}; color: {INK}; border: 1.5px solid {EDGE}; box-shadow: 0 3px 0 {EDGE};
                      transition: transform .08s ease, box-shadow .08s ease, background-color .15s ease; }}
.stButton > button p {{ font-stretch: 112%; font-weight: 650; font-size: .95rem; letter-spacing: .01em; }}
.stButton > button:hover {{ background: {SKY}; border-color: {SKY_D}; color: {SKY_D}; }}
.stButton > button:active {{ transform: translateY(2px); box-shadow: 0 1px 0 {EDGE}; }}
.stButton > button:focus-visible {{ outline: 3px solid {SKY_D}; outline-offset: 2px; }}
.stButton > button[kind="primary"] {{ background: {BTN}; border-color: {BTN_EDGE}; color: #FFFFFF; box-shadow: 0 3px 0 {BTN_SHADOW}; }}
.stButton > button[kind="primary"]:hover {{ background: {BTN_EDGE}; border-color: {BTN_EDGE}; color: #FFFFFF; }}
.stButton > button[kind="primary"]:active {{ box-shadow: 0 1px 0 {BTN_SHADOW}; }}
@media (prefers-reduced-motion: reduce) {{ .stButton > button {{ transition: none; }} .stButton > button:active {{ transform: none; }} }}
[data-baseweb="tab-list"] button p {{ font-size: 1rem; font-weight: 600; }}

@media (max-width: 640px) {{
    .ap-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data helpers
@st.cache_data(ttl=600, show_spinner=False)
def get_properties():
    r = requests.get(f"{API_BASE_URL}/properties", timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def get_prediction(property_id):
    r = requests.post(f"{API_BASE_URL}/predict", json={"property_index": property_id}, timeout=60)
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(data.get("error", f"HTTP {r.status_code}"))
    return data


def get_explanation(property_id):
    r = requests.post(f"{API_BASE_URL}/explain_prediction", json={"property_index": property_id}, timeout=240)
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(data.get("error", f"HTTP {r.status_code}"))
    return data


def money(v):
    return f"${v:,.0f}"


def signed(v):
    return ("+" if v >= 0 else "−") + money(abs(v))


def short_money(v):
    return f"${v / 1e6:,.2f}M" if v >= 1e6 else f"${v / 1e3:,.0f}K"


def brightened(data_uri):
    """Sentinel-2 patches are stored dark; stretch them so the ground is visible."""
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype="float32")
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.85
    return (img * 255).astype("uint8")


def png_data_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def style(fig, height, legend=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), font=dict(family=FONT, color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=WHITE, showlegend=legend,
                      title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"))
    return fig


def chart(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


def howto(html):
    st.markdown(f'<div class="ap-howto"><b>How to read this:</b> {html}</div>', unsafe_allow_html=True)


def place(r, c, g):
    rows = ["top", "upper-middle", "lower-middle", "bottom"]
    cols = ["left", "centre-left", "centre-right", "right"]
    return f"{rows[min(3, r * 4 // g)]} {cols[min(3, c * 4 // g)]}"


FACT_UNITS = {
    "livingAreaSqFt": ("Living area", lambda v: f"{v:,.0f} sq ft"),
    "numOfBedrooms": ("Bedrooms", lambda v: f"{v:g}"),
    "numOfBathrooms": ("Bathrooms", lambda v: f"{v:g}"),
    "yearBuilt": ("Year built", lambda v: f"{v:.0f}"),
    "lotSizeSqFt": ("Lot size", lambda v: f"{v:,.0f} sq ft"),
    "avgSchoolRating": ("School rating", lambda v: f"{v:.1f} / 10"),
}
LABEL_TO_KEY = {label: key for key, (label, _) in FACT_UNITS.items()}


# ------------------------------------------------------------------ header + controls
_, theme_col = st.columns([6, 1])
with theme_col:
    st.toggle("Dark mode", key="dark_mode")

st.markdown(f"""
<div class="ds-head">
  <svg class="ds-mark" width="72" height="44" viewBox="0 0 72 44" aria-hidden="true">
    <defs><clipPath id="ds-left"><circle cx="27" cy="22" r="17"/></clipPath></defs>
    <circle cx="27" cy="22" r="17" fill="{SKY}"/>
    <circle cx="45" cy="22" r="17" fill="{MINT}"/>
    <circle cx="45" cy="22" r="17" fill="{LAVENDER}" clip-path="url(#ds-left)"/>
    <circle cx="27" cy="22" r="17" fill="none" stroke="{SKY_D}" stroke-width="2"/>
    <circle cx="45" cy="22" r="17" fill="none" stroke="{MINT_D}" stroke-width="2"/>
  </svg>
  <div class="ds-word">DualSight</div>
  <div class="ds-tag">Austin home values seen two ways at once: from space, through Sentinel-2 satellite imagery,
  and on paper, through the listing details. Every estimate shows what drove it.</div>
  <div class="ds-flow">
    <span class="pill" style="background:{SKY}; color:{SKY_D}">Satellite view</span><span class="op">+</span>
    <span class="pill" style="background:{MINT}; color:{MINT_D}">Listing details</span><span class="op">=</span>
    <span class="pill" style="background:{LAVENDER}; color:{LAVENDER_D}">One explained estimate</span>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    properties = get_properties()
except Exception:
    st.error(f"Can't reach the DualSight backend at {API_BASE_URL}. Start it with `python backend/main.py` in the project "
             "folder, wait for \"Active model\" in that window, then refresh this page.")
    st.stop()

options = {p["id"]: f"{p['streetAddress']}, {p['zipcode']}" for p in properties}
ids = list(options)
ss = st.session_state
ss.setdefault("shown", None)
ss.setdefault("explanations", {})
ss.setdefault("open_explain", set())
ss.setdefault("open_ratings", set())


def show_selected():
    ss.shown = ss.pid


def surprise():
    ss.pid = random.choice(ids)
    ss.shown = ss.pid


_, c_pick, c_show, c_rand, _ = st.columns([.3, 2.3, 1, 1, .3], vertical_alignment="bottom")
with c_pick:
    st.selectbox(f"Choose a home ({len(ids)} Austin listings, type to search)", ids, format_func=options.get, key="pid")
with c_show:
    st.button("Show prediction", type="primary", on_click=show_selected, width="stretch")
with c_rand:
    st.button("Surprise me", on_click=surprise, width="stretch")

if ss.shown is None:
    st.markdown('<div class="ap-empty">Pick an address above, then press <b>Show prediction</b>.<br>'
                'Or press <b>Surprise me</b> for a random Austin home.</div>', unsafe_allow_html=True)
    st.stop()

property_id = ss.shown
if ss.pid != property_id:
    st.info(f"Showing **{options[property_id]}**. Press Show prediction to estimate **{options[ss.pid]}** instead.")

try:
    with st.spinner("Estimating…"):
        pred = get_prediction(property_id)
except Exception as e:
    st.error(f"The backend couldn't estimate this home: {e}")
    st.stop()

facts = pred.get("facts") or {}
predicted = float(pred.get("predicted_price") or 0)
actual = float(pred["actual_price"]) if pred.get("actual_price") else None
street, _, zipcode = options[property_id].rpartition(", ")

# ------------------------------------------------------------------ plate + estimate
st.write("")
left, right = st.columns([1.05, 1], gap="large")
with left:
    img_html = f'<img src="{png_data_uri(brightened(pred["satellite_image"]))}" alt="Satellite view around {street}">' \
        if pred.get("satellite_image") else ""
    lat, lon = facts.get("latitude"), facts.get("longitude")
    coords = f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.4f}° {'W' if lon < 0 else 'E'}" if lat and lon else ""
    st.markdown(f"""<div class="ap-plate">{img_html}
      <div class="ap-plate-caption"><span>Sentinel-2 true colour, 2020, brightened for display</span><span>{coords}</span></div>
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown(f'<p class="ap-address">{street}</p><p class="ap-zip">Austin, TX {zipcode}</p>', unsafe_allow_html=True)
    est_html = f'<div class="ap-estimate"><p class="ap-label">Estimated value</p><div class="ap-price">{money(predicted)}</div>'
    if actual:
        lo, hi = min(actual, predicted) * 0.85, max(actual, predicted) * 1.12
        pos = lambda v: (v - lo) / (hi - lo) * 100
        gap_pct = (predicted - actual) / actual * 100
        deep, pale = (MINT_D, MINT) if gap_pct >= 0 else (ROSE_D, ROSE)
        a, p = pos(actual), pos(predicted)
        est_html += f"""<p class="ap-listed">Listed at {money(actual)}</p>
        <div class="ap-scale" role="img" aria-label="Estimate {money(predicted)} against listing {money(actual)}">
          <div class="ap-scale-track"></div>
          <div class="ap-scale-gap" style="left:{min(a, p):.2f}%; width:{abs(a - p):.2f}%; background:{UP_BAR if gap_pct >= 0 else DOWN_BAR};"></div>
          <div class="ap-tick est" style="left:{p:.2f}%">Estimate</div>
          <div class="ap-tick lst" style="left:{a:.2f}%">Listed</div>
        </div>
        <div class="ap-scale-ends"><span>{short_money(lo)}</span><span>{short_money(hi)}</span></div>
        <span class="ap-chip" style="background:{pale}; color:{deep}">{abs(gap_pct):.1f}% {'above' if gap_pct >= 0 else 'below'}
        the listing ({signed(predicted - actual)})</span>"""
    est_html += "</div>"
    st.markdown(est_html, unsafe_allow_html=True)

    tile_colours = [(MINT, MINT_D), (SKY, SKY_D), (LAVENDER, LAVENDER_D), (PEACH, PEACH_D), (BUTTER, BUTTER_D), (ROSE, ROSE_D)]
    icons = [icon(n, 15) for n in ("area", "bed", "bath", "calendar", "tree", "school")]
    tiles = ""
    for (key, (label, fmt)), (bg, fg), ic in zip(FACT_UNITS.items(), tile_colours, icons):
        v = facts.get(key)
        tiles += f'<div class="ap-tile" style="background:{bg}"><span style="color:{fg}">{ic}{label}</span>' \
                 f'<b>{fmt(v) if v is not None else "—"}</b></div>'
    st.markdown(f'<div class="ap-tiles">{tiles}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="ap-chip" style="background:{LAVENDER}; color:{LAVENDER_D}">Model: {pred.get("model_name", "unknown")}</span>',
                unsafe_allow_html=True)


def toggle(name, pid):
    ss[name].symmetric_difference_update({pid})
    # Closing an explanation that had no image map lets the next "Explain" fetch a fresh one
    if name == "open_explain" and pid not in ss[name] and (ss.explanations.get(pid) or {}).get("occlusion") is None:
        ss.explanations.pop(pid, None)


st.write("")
b1, b2, _ = st.columns([1.2, 1.4, 2.4])
explain_open = property_id in ss.open_explain
ratings_open = property_id in ss.open_ratings
with b1:
    st.button("Hide explanation" if explain_open else "Explain this estimate", type="primary",
              on_click=toggle, args=("open_explain", property_id), width="stretch")
with b2:
    st.button("Hide ratings & amenities" if ratings_open else "Ratings & amenities",
              on_click=toggle, args=("open_ratings", property_id), width="stretch")

# ------------------------------------------------------------------ explanation
if property_id in ss.open_explain:
    if property_id not in ss.explanations:
        with st.spinner("Working out what drove this estimate. This takes a few seconds…"):
            try:
                ss.explanations[property_id] = get_explanation(property_id)
            except Exception as e:
                st.error(f"The explanation failed: {e}")
    exp = ss.explanations.get(property_id)
else:
    exp = None

if exp:
    est = float(exp["predicted_price"])
    steps = exp.get("steps") or [{"label": k, "dollars": v} for k, v in exp.get("contributions", {}).items()]
    base = float(exp.get("base_price", 0))
    small = max(500.0, 0.005 * est)
    ups = [s for s in steps if s["dollars"] >= small][:2]
    downs = [s for s in steps if s["dollars"] <= -small][:2]
    phrase = lambda items, verb: " and ".join(f"{s['label'].lower()} {verb} {money(abs(s['dollars']))}" for s in items)
    parts = [x for x in (phrase(ups, "adds"), phrase(downs, "takes off")) if x]

    st.markdown('<div class="ap-section">Why this estimate</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ap-lead">The model starts from a typical Austin home at <b>{money(base)}</b>. For {street}, '
                f'{"; ".join(parts) if parts else "no factor moves the price much"}, arriving at <b>{money(est)}</b>.</p>',
                unsafe_allow_html=True)

    t_factors, t_image, t_accuracy, t_models = st.tabs(["Price factors", "Satellite image", "Accuracy", "Model comparison"])

    # ---- factors
    with t_factors:
        shown = [s for s in steps if abs(s["dollars"]) >= small]
        rest = sum(s["dollars"] for s in steps if abs(s["dollars"]) < small)
        if abs(rest) >= small:
            shown.append({"label": "Other small factors", "dollars": rest})
        elif rest and shown:
            k = min(range(len(shown)), key=lambda j: abs(shown[j]["dollars"]))
            shown[k] = {**shown[k], "dollars": shown[k]["dollars"] + rest}

        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            labels = ["Typical home"] + [s["label"] for s in shown] + ["This estimate"]
            values = [base] + [s["dollars"] for s in shown] + [est]
            running = np.cumsum([base] + [s["dollars"] for s in shown])
            fig = go.Figure(go.Waterfall(
                orientation="h", y=labels, x=values, measure=["absolute"] + ["relative"] * len(shown) + ["total"],
                text=[money(base)] + [signed(s["dollars"]) for s in shown] + [money(est)], textposition="outside",
                cliponaxis=False, increasing=dict(marker=dict(color=UP_BAR)), decreasing=dict(marker=dict(color=DOWN_BAR)),
                totals=dict(marker=dict(color=BASE_BAR)), connector=dict(line=dict(color=RULE, width=1)),
                hovertemplate="%{y}: %{text}<extra></extra>"))
            style(fig, 80 + 44 * len(labels))
            fig.update_layout(title_text="From a typical home to this estimate", margin=dict(l=10, r=90, t=36, b=10))
            lo_, hi_ = float(min(running.min(), est)), float(max(running.max(), est))
            pad = (hi_ - lo_) * 0.25 + 1
            fig.update_xaxes(tickprefix="$", tickformat="~s", gridcolor=RULE, zeroline=False, range=[max(0, lo_ - pad), hi_ + pad])
            fig.update_yaxes(automargin=True, autorange="reversed")
            chart(fig)
            howto(f"Start at the <b>blue bar on top</b>, the price of a typical Austin home in this model. Each <b>green bar</b> "
                  f"adds money and each <b>pink bar</b> takes it away, biggest first. The <b>blue bar at the bottom</b> is this "
                  f"home's estimate. The axis is zoomed in, so it doesn't start at $0.")
        with c2:
            weights = [abs(s["dollars"]) for s in shown]
            palette = [MINT, SKY, BUTTER, PEACH, LAVENDER, ROSE, EXTRA1, EXTRA2]
            fig = go.Figure(go.Pie(labels=[s["label"] for s in shown], values=weights, hole=.58, sort=True,
                                   marker=dict(colors=palette[:len(shown)], line=dict(color=WHITE, width=2)),
                                   textinfo="percent", textfont=dict(color=INK),
                                   hovertemplate="%{label}: %{percent} of the total movement<extra></extra>"))
            style(fig, 360, legend=True)
            fig.update_layout(title_text="Share of influence", legend=dict(orientation="h", y=-0.05, font=dict(size=12, color=INK)),
                              annotations=[dict(text=f"<b>{money(sum(weights))}</b><br>moved", showarrow=False,
                                                font=dict(size=14, color=INK))])
            chart(fig)
            howto("Each slice is one factor's share of all the money moved, whether up or down. "
                  "A bigger slice means that factor had more say in this estimate.")

        st.markdown("##### Factor by factor")
        typical = exp.get("typical_facts") or {}
        cards = ""
        for s in shown:
            d = s["dollars"]
            pale, deep = (MINT, MINT_D) if d >= 0 else (ROSE, ROSE_D)
            key = LABEL_TO_KEY.get(s["label"])
            if s["label"] == "Satellite image":
                body = ("What the neighbourhood looks like from space: greenery, roofs, roads and density. "
                        f"Compared with a typical satellite patch, it {'raises' if d >= 0 else 'lowers'} the price.")
            elif key and facts.get(key) is not None and typical.get(key) is not None:
                v, t = facts[key], typical[key]
                fmt = FACT_UNITS[key][1]
                rel = "higher than" if v > t else "lower than" if v < t else "the same as"
                body = (f"This home: <b>{fmt(v)}</b>, typical: {fmt(t)}. It's {rel} typical, and the model "
                        f"{'raises' if d >= 0 else 'lowers'} the price.")
            else:
                body = "Several factors that each moved the price only slightly, grouped together."
            cards += (f'<div class="ap-factor" style="background:{pale}"><div class="ttl"><b>{s["label"]}</b></div>'
                      f'<div class="amt" style="color:{deep}">{signed(d)}</div><p>{body}</p></div>')
        st.markdown(f'<div class="ap-factors">{cards}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="ap-note" style="margin-top:.8rem">Method: {exp.get("method", "")} The model learned these effects '
                    'from data, so they can sometimes look surprising, such as an extra bedroom lowering the price when '
                    'living area stays the same.</p>', unsafe_allow_html=True)

    # ---- satellite image
    with t_image:
        grid = exp.get("occlusion")
        if grid and pred.get("satellite_image"):
            grid = np.array(grid, dtype="float64")
            g = grid.shape[0]
            dollars = est * (1 - np.exp(-grid))          # what each square adds to the estimate
            img = brightened(pred["satellite_image"])
            cell = img.shape[0] / g
            lim = max(float(np.abs(dollars).max()), 1.0)
            order = np.argsort(dollars.ravel())
            reds = [divmod(int(i), g) for i in order[:3] if dollars.ravel()[i] < 0]
            greens = [divmod(int(i), g) for i in order[::-1][:3] if dollars.ravel()[i] > 0]

            c1, c2 = st.columns([1.35, 1], gap="large")
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Image(z=img, hoverinfo="skip"))
                fig.add_trace(go.Heatmap(
                    z=dollars, x0=cell / 2, dx=cell, y0=cell / 2, dy=cell, zmid=0, zmin=-lim, zmax=lim, opacity=.6,
                    colorscale=[[0, DOWN_STRONG], [.25, DOWN_BAR], [.5, "rgba(255,255,255,0.04)"], [.75, UP_BAR], [1, UP_STRONG]],
                    colorbar=dict(title=dict(text="Adds to<br>the price", side="top"), tickprefix="$", tickformat="~s",
                                  thickness=12, outlinewidth=0),
                    hovertemplate="This square adds %{z:$,.0f} to the estimate<extra></extra>"))
                for n, (r, c) in enumerate(greens, 1):
                    fig.add_annotation(x=(c + .5) * cell, y=(r + .5) * cell, text=f"<b>G{n}</b>", showarrow=False,
                                       bgcolor=MINT, bordercolor=MINT_D, borderwidth=1.5, font=dict(color=MINT_D, size=12))
                for n, (r, c) in enumerate(reds, 1):
                    fig.add_annotation(x=(c + .5) * cell, y=(r + .5) * cell, text=f"<b>R{n}</b>", showarrow=False,
                                       bgcolor=ROSE, bordercolor=ROSE_D, borderwidth=1.5, font=dict(color=ROSE_D, size=12))
                style(fig, 540)
                fig.update_layout(title_text="Which parts of the satellite image moved the price")
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False, scaleanchor="x")
                chart(fig)
                howto(f"The patch is split into a {g}×{g} grid. Each square was greyed out in turn and the home "
                      "re-estimated. <b>Green squares</b> are areas whose contents add value, because hiding them lowers "
                      "the estimate. <b>Pink squares</b> take value away, because hiding them raises it. Clear squares hardly "
                      "matter. The labels mark the three strongest of each.")
            with c2:
                gl = "".join(f'<li><span class="ap-tag" style="background:{MINT};color:{MINT_D}">G{n}</span>'
                             f'{place(r, c, g).capitalize()}: adds <b>{money(dollars[r, c])}</b></li>'
                             for n, (r, c) in enumerate(greens, 1))
                rl = "".join(f'<li><span class="ap-tag" style="background:{ROSE};color:{ROSE_D}">R{n}</span>'
                             f'{place(r, c, g).capitalize()}: takes off <b>{money(-dollars[r, c])}</b></li>'
                             for n, (r, c) in enumerate(reds, 1))
                img_step = next((s["dollars"] for s in steps if s["label"] == "Satellite image"), None)
                st.markdown(f"""
                <div class="ap-card" style="background:{MINT}; border-color:{MINT}"><b style="color:{MINT_D}"><span class="ap-swatch" style="background:{UP_BAR}"></span>Green squares raise the price</b>
                  <ul class="ap-squares">{gl or '<li>No area raises the price noticeably.</li>'}</ul></div>
                <div class="ap-card" style="background:{ROSE}; border-color:{ROSE}"><b style="color:{ROSE_D}"><span class="ap-swatch" style="background:{DOWN_BAR}"></span>Pink squares lower the price</b>
                  <ul class="ap-squares">{rl or '<li>No area lowers the price noticeably.</li>'}</ul></div>
                <p class="ap-note">Looking at each square on its own, the most valuable area is worth {money(dollars.max())}
                and the most costly {money(-dollars.min())}. Squares interact, so these amounts don't add up to the image's
                overall effect{f" ({signed(img_step)} on the Price factors tab)" if img_step is not None else ""}.</p>
                <p class="ap-note">The model only sees textures and colours: it has no labels for parks or roads. Compare the
                squares with the satellite photo to see what they contain.</p>""", unsafe_allow_html=True)

            vals = np.sort(dollars.ravel())[::-1]
            fig = go.Figure(go.Bar(x=list(range(1, len(vals) + 1)), y=vals,
                                   marker_color=[UP_BAR if v >= 0 else DOWN_BAR for v in vals],
                                   hovertemplate="Square ranked %{x}: %{y:$,.0f}<extra></extra>"))
            style(fig, 260)
            fig.update_layout(title_text=f"All {g * g} squares, from most valuable to most costly", bargap=.15)
            fig.update_xaxes(title="Squares, ranked", showticklabels=False)
            fig.update_yaxes(tickprefix="$", tickformat="~s", gridcolor=RULE, zeroline=True, zerolinecolor=MUTED)
            chart(fig)
            howto("One bar per square, sorted. Bars above the line add value and bars below take it away. A few tall bars "
                  "mean a handful of areas drive the image's effect, while many short bars mean the influence is spread out.")
        else:
            st.info("The image map isn't available for this estimate. With the deep + boosting model it needs ImageNet "
                    "weights, which the backend downloads the first time (about 94 MB). Check the backend window, then "
                    "press Hide explanation and then Explain this estimate again.")
            if exp.get("occlusion_note"):
                st.caption(exp["occlusion_note"])

    # ---- accuracy
    with t_accuracy:
        held = exp.get("held_out")
        if held:
            act, prd = np.array(held["actual"]), np.array(held["predicted"])
            signed_err = (prd - act) / act * 100
            err = np.abs(signed_err)
            this_err = (est - actual) / actual * 100 if actual else None
            st.markdown(f'<p class="ap-lead">Tested on <b>{len(act):,}</b> homes the model never trained on, the typical estimate '
                        f'is off by <b>{np.median(err):.1f}%</b>, and <b>{np.mean(err <= 20) * 100:.0f}%</b> of estimates land '
                        f'within 20% of the listing price.'
                        + (f" This home's estimate is {abs(this_err):.1f}% {'above' if this_err >= 0 else 'below'} its listing."
                           if this_err is not None else "") + '</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                top = float(np.percentile(np.concatenate([act, prd]), 99.5))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, top], y=[0, top * 1.2], mode="lines", line=dict(width=0), hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[0, top], y=[0, top * .8], mode="lines", line=dict(width=0), fill="tonexty",
                                         fillcolor=rgba(UP_BAR, .18), hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=act, y=prd, mode="markers", marker=dict(size=5, color=SKY_D, opacity=DOT_ALPHA),
                                         hovertemplate="Listed %{x:$,.0f}<br>Estimated %{y:$,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=INK, width=1, dash="dot"),
                                         hoverinfo="skip"))
                if actual:
                    fig.add_trace(go.Scatter(x=[actual], y=[est], mode="markers",
                                             marker=dict(symbol="star", size=22, color=STAR, line=dict(color=INK, width=1.2)),
                                             hovertemplate=f"{street}<br>Listed %{{x:$,.0f}}<br>Estimated %{{y:$,.0f}}<extra></extra>"))
                    fig.add_annotation(x=actual, y=est, text="This home", showarrow=True, arrowhead=0, ax=-60, ay=-70,
                                       arrowcolor=INK, bgcolor=BUTTER, bordercolor=BUTTER_D, borderwidth=1, borderpad=4)
                style(fig, 470)
                fig.update_layout(title_text="Estimated vs listed price")
                fig.update_xaxes(range=[0, top], title="Listed price", tickprefix="$", tickformat="~s", gridcolor=RULE, zeroline=False)
                fig.update_yaxes(range=[0, top], title="Estimated price", tickprefix="$", tickformat="~s", gridcolor=RULE, zeroline=False)
                chart(fig)
                howto("Each dot is one unseen home. Dots on the <b>dotted line</b> are perfect estimates, dots above it are "
                      "too high and dots below are too low. The <b>green band</b> is within 20% of the listing. "
                      "The star is this home" + ("." if exp.get("in_test_set") else
                                              ", which wasn't in this test set, so the model may have seen it in training."))
            with c2:
                clipped = np.clip(signed_err, -100, 100)
                fig = go.Figure(go.Histogram(x=clipped, xbins=dict(start=-100, end=100, size=5),
                                             marker=dict(color=LAVENDER, line=dict(color=LAVENDER_D, width=1)),
                                             hovertemplate="%{x}% off: %{y} homes<extra></extra>"))
                fig.add_vrect(x0=-20, x1=20, fillcolor=MINT, opacity=.45, line_width=0, layer="below")
                if this_err is not None:
                    fig.add_vline(x=float(np.clip(this_err, -100, 100)), line=dict(color=INK, width=2, dash="dash"))
                    fig.add_annotation(x=float(np.clip(this_err, -100, 100)), y=1, yref="paper", text=f"This home {this_err:+.0f}%",
                                       showarrow=False, yanchor="bottom", bgcolor=BUTTER, bordercolor=BUTTER_D, borderwidth=1)
                style(fig, 470)
                fig.update_layout(title_text="How far off the estimates are", bargap=.05)
                fig.update_xaxes(title="Estimate vs listing (%)", ticksuffix="%", gridcolor=RULE, range=[-100, 100])
                fig.update_yaxes(title="Number of homes", gridcolor=RULE)
                chart(fig)
                howto("Bars to the left of 0% are under-estimates, bars to the right are over-estimates. The <b>green zone</b> is "
                      "within 20%. A tall, narrow pile around 0% means the model is usually close. Errors beyond ±100% are "
                      "grouped at the edges.")

            edges = np.percentile(act, [0, 20, 40, 60, 80, 100])
            band = np.clip(np.searchsorted(edges, act, side="right") - 1, 0, 4)
            med = [float(np.median(err[band == b])) for b in range(5)]
            bias = [float(np.median(signed_err[band == b])) for b in range(5)]
            names = [f"{short_money(edges[b])}–{short_money(edges[b + 1])}" for b in range(5)]
            fig = go.Figure(go.Bar(x=names, y=med, marker_color=[MINT, SKY, BUTTER, PEACH, ROSE],
                                   marker_line=dict(color=[MINT_D, SKY_D, BUTTER_D, PEACH_D, ROSE_D], width=1.5),
                                   text=[f"{m:.1f}%<br><span style='font-size:11px'>usually {'over' if b > 0 else 'under'} by {abs(b):.0f}%</span>"
                                         for m, b in zip(med, bias)], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}: typical error %{y:.1f}%<extra></extra>"))
            style(fig, 340)
            fig.update_layout(title_text="Typical error by price range (fifths of the test homes)")
            fig.update_yaxes(title="Typical error", ticksuffix="%", gridcolor=RULE, range=[0, max(med) * 1.45])
            chart(fig)
            worst = int(np.argmax(med))
            howto(f"Each bar is one fifth of the test homes, from cheapest to priciest. Its height is the typical error, and "
                  f"the note says whether estimates there tend to run high or low. The model struggles most in the "
                  f"<b>{names[worst]}</b> range ({med[worst]:.1f}%), where it usually "
                  f"{'over' if bias[worst] > 0 else 'under'}-estimates.")
        else:
            st.info("No held-out results for this model yet. Run `python pipeline/evaluate_model.py` once, then explain again.")

    # ---- model comparison
    with t_models:
        comp = exp.get("model_comparison")
        if comp:
            comp = sorted(comp, key=lambda m: m["median_pct_error"])
            c1, c2 = st.columns(2, gap="large")
            colours = {"boosting_hybrid": (MINT, MINT_D), "neural_hybrid": (SKY, SKY_D),
                       "numbers_only": (BUTTER, BUTTER_D), "image_only": (PEACH, PEACH_D)}
            labels = [m["label"] + (" (in use)" if m["active"] else "") for m in comp]
            with c1:
                fig = go.Figure(go.Bar(y=labels, x=[m["median_pct_error"] for m in comp], orientation="h",
                                       marker_color=[colours.get(m["kind"], (LAVENDER, LAVENDER_D))[0] for m in comp],
                                       marker_line=dict(color=[colours.get(m["kind"], (LAVENDER, LAVENDER_D))[1] for m in comp], width=1.5),
                                       text=[f"{m['median_pct_error']:.1f}%" for m in comp], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: typical error %{x:.1f}%<extra></extra>"))
                style(fig, 320)
                fig.update_layout(title_text="Typical error (lower is better)", margin=dict(l=10, r=50, t=36, b=10))
                fig.update_xaxes(ticksuffix="%", gridcolor=RULE)
                fig.update_yaxes(autorange="reversed", automargin=True)
                chart(fig)
            with c2:
                fig = go.Figure(go.Bar(y=labels, x=[m["within_20"] for m in comp], orientation="h",
                                       marker_color=[colours.get(m["kind"], (LAVENDER, LAVENDER_D))[0] for m in comp],
                                       marker_line=dict(color=[colours.get(m["kind"], (LAVENDER, LAVENDER_D))[1] for m in comp], width=1.5),
                                       text=[f"{m['within_20']:.0f}%" for m in comp], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x:.0f}% within 20%<extra></extra>"))
                style(fig, 320)
                fig.update_layout(title_text="Estimates within 20% of the listing (higher is better)", margin=dict(l=10, r=50, t=36, b=10))
                fig.update_xaxes(ticksuffix="%", gridcolor=RULE, range=[0, 100])
                fig.update_yaxes(autorange="reversed", automargin=True, showticklabels=False)
                chart(fig)
            by_kind = {m["kind"]: m for m in comp}
            extra = ""
            if "numbers_only" in by_kind and "boosting_hybrid" in by_kind:
                extra = (f" Adding the satellite image to the listing numbers cuts the typical error from "
                         f"{by_kind['numbers_only']['median_pct_error']:.1f}% to {by_kind['boosting_hybrid']['median_pct_error']:.1f}%, "
                         "so the view from space carries real information.")
            howto(f"All models were scored on the same {comp[0]['n']:,} unseen homes. The original project used the "
                  f"<b>image-only</b> model; the two hybrids also read the listing details.{extra}")
        else:
            st.info("Model comparison appears after importing the Kaggle models and running `python pipeline/evaluate_model.py`.")

# ------------------------------------------------------------------ ratings & amenities
if property_id in ss.open_ratings:
    st.markdown('<div class="ap-section">Neighbourhood ratings and amenities</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="ap-chip" style="background:{BUTTER}; color:{BUTTER_D}">Rule-of-thumb formulas on the '
                f'listing details, not predictions from the model</span>', unsafe_allow_html=True)
    scores = {"Neighbourhood": pred.get("neighborhood_score", 0), "Accessibility": pred.get("accessibility_score", 0),
              "Green space": pred.get("green_space_score", 0)}
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        cats = list(scores) + [list(scores)[0]]
        vals = list(scores.values()) + [list(scores.values())[0]]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[5.5] * 4, theta=cats, fill="toself", fillcolor=rgba(LAVENDER_D, .18),
                                      line=dict(color=LAVENDER_D, dash="dot", width=1), name="Middle of the scale",
                                      hoverinfo="skip"))
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", fillcolor=rgba(UP_BAR, .45),
                                      line=dict(color=MINT_D, width=2), name="This home",
                                      hovertemplate="%{theta}: %{r:.1f} / 10<extra></extra>"))
        style(fig, 380, legend=True)
        fig.update_layout(title_text="Ratings at a glance", legend=dict(orientation="h", y=-0.08, font=dict(color=INK)),
                          margin=dict(l=60, r=60, t=50, b=10),
                          polar=dict(bgcolor=WHITE, radialaxis=dict(range=[0, 10], gridcolor=RULE, tickfont=dict(size=10)),
                                     angularaxis=dict(gridcolor=RULE, rotation=90, direction="clockwise")))
        chart(fig)
    with c2:
        g1, g2, g3 = st.columns(3)
        for col, (name, v), (pale, deep) in zip((g1, g2, g3), scores.items(),
                                                ((SKY, SKY_D), (PEACH, PEACH_D), (MINT, MINT_D))):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=float(v), number=dict(suffix=" / 10", valueformat=".1f", font=dict(size=24, color=INK)),
                gauge=dict(axis=dict(range=[0, 10], tickwidth=1, tickcolor=MUTED), bar=dict(color=deep, thickness=.3),
                           bgcolor=WHITE, borderwidth=0,
                           steps=[dict(range=[0, 4], color=LOW_STEP), dict(range=[4, 7], color=pale), dict(range=[7, 10], color=pale)])))
            style(fig, 210)
            fig.update_layout(title_text=name, margin=dict(l=32, r=32, t=40, b=0))
            with col:
                chart(fig)
        s_ = facts.get("avgSchoolRating") or 0
        la = facts.get("livingAreaSqFt") or 0
        bd, bt = facts.get("numOfBedrooms") or 0, facts.get("numOfBathrooms") or 0
        lot, yb = facts.get("lotSizeSqFt") or 0, facts.get("yearBuilt") or 0
        cap = lambda x: f"{x:.1f}" + (" → capped at 10" if x > 10 else " → raised to 1" if x < 1 else "")
        st.markdown(f"""<div class="ap-howto"><b>How each rating is worked out</b> (from the original project, all kept between 1 and 10):<br>
        <b>Neighbourhood</b> = 2.5 × school rating + living area ÷ 1,000 = 2.5 × {s_:.1f} + {la:,.0f} ÷ 1,000 = {cap(2.5 * s_ + la / 1000)}<br>
        <b>Accessibility</b> = 1.5 × (bedrooms + bathrooms) = 1.5 × ({bd:g} + {bt:g}) = {cap(1.5 * (bd + bt))}<br>
        <b>Green space</b> = lot ÷ 1,000 + (year built − 1990) ÷ 10 = {lot:,.0f} ÷ 1,000 + ({yb:.0f} − 1990) ÷ 10 = {cap(lot / 1000 + (yb - 1990) / 10)}
        </div>""", unsafe_allow_html=True)

    pct = float(pred.get("price_percentile") or 0)
    fig = go.Figure(go.Bar(x=[100], y=[""], orientation="h", marker_color=RULE, hoverinfo="skip"))
    fig.add_trace(go.Bar(x=[pct], y=[""], orientation="h", marker_color=PEACH, marker_line=dict(color=PEACH_D, width=1.5),
                         text=[f"{pct:.0f}%"], textposition="inside", hoverinfo="skip"))
    style(fig, 120)
    fig.update_layout(barmode="overlay", title_text="Where the estimate sits between $200K and $800K",
                      margin=dict(l=10, r=10, t=36, b=10))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=RULE)
    chart(fig)
    howto(f"The amenity guesses below use this position: the estimate of {money(predicted)} is {pct:.0f}% of the way from "
          "$200K to $800K. Pricier homes are assumed to sit closer to schools, healthcare and the city centre.")

    st.markdown("##### Nearby amenities (guessed)")
    icons = {"schools": "school", "healthcare": "health", "city_center": "city", "parks": "tree", "public_transport": "bus"}
    tints = [(SKY, SKY_D), (PEACH, PEACH_D), (LAVENDER, LAVENDER_D), (MINT, MINT_D), (BUTTER, BUTTER_D)]
    level = {"high": 3, "moderate": 2, "low": 1}
    cards = ""
    for (name, d), (pale, deep) in zip((pred.get("inferred_amenities") or {}).items(), tints * 2):
        n = level.get(d.get("confidence", "low"), 1)
        dots = "".join(f'<span style="background:{deep if i < n else WHITE}; border:1px solid {deep}"></span>' for i in range(3))
        cards += (f'<div class="ap-amenity" style="background:{pale}"><span class="badge" style="color:{deep}">'
                  f'{icon(icons.get(name, "pin"), 24)}</span>'
                  f'<div class="ttl"><b>{name.replace("_", " ").capitalize()}</b></div><p><b>{d.get("proximity", "").capitalize()}</b></p>'
                  f'<p class="ap-dots">{dots} <span style="width:auto;height:auto;color:{deep};font-size:.8rem">'
                  f'{d.get("confidence", "low")} confidence</span></p>'
                  f'<p style="color:{MUTED}">{d.get("reasoning", "")}</p></div>')
    st.markdown(f'<div class="ap-amenities">{cards or "<p>No amenity guesses for this home.</p>"}</div>', unsafe_allow_html=True)
    st.markdown('<p class="ap-note" style="margin-top:.8rem">These guesses come from simple rules in the original project '
                '(price range and ratings), not from maps or the model. Treat them as a talking point, not a fact.</p>',
                unsafe_allow_html=True)
