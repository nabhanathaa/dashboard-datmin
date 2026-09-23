# =====================================================================================
#  DASHBOARD KLASTERISASI RISIKO DBD — PROVINSI JAWA BARAT 2025
#  Kelompok 5 — Data Mining, Politeknik Statistika STIS
#  Jalankan:  streamlit run app.py
#  Data (folder ./data):
#   - Master_Dataset_DBD_Jabar_2025.xlsx        (sheet "Data")
#   - Hasil_Klasterisasi_DBD_Jabar_2025.xlsx    (ekspor notebook v3, folder output_analisis)
#   - klaster_jabar_2025.geojson                (ekspor notebook) atau FIX_KABKOTA_JABAR.shp
# =====================================================================================
import os, json, itertools, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score)

_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_stis.png")
try:
    st.set_page_config(page_title="Klasterisasi Risiko DBD Jawa Barat 2025",
                       page_icon=_ICON if os.path.exists(_ICON) else ":material/coronavirus:",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:                                    # versi Streamlit lama
    st.set_page_config(page_title="Klasterisasi Risiko DBD Jawa Barat 2025",
                       layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ sistem desain
if "dark" not in st.session_state: st.session_state.dark = False

PALETTE = {
    "light": dict(BG="#FFFFFF", SURF="#FFFFFF", SOFT="#F3F8F5", INK="#10251C", MUTED="#5C7065", LINE="#DCE6E0",
                  PLOT_BG="#FAFCFB", HERO_TXT="#FFFFFF"),
    "dark":  dict(BG="#0E1A16", SURF="#14231D", SOFT="#18302A", INK="#E8F1EC", MUTED="#9FB6AC", LINE="#24403A",
                  PLOT_BG="#14231D", HERO_TXT="#FFFFFF"),
}
GREEN, GREEN_D, GREEN_L = "#17784C", "#0E4F33", "#2E9E6B"
BLUE, BLUE_D, BLUE_L = "#1B4F8A", "#0E3462", "#4F86C6"
RISK, WARN, INFO = "#C0392B", "#D98032", "#1B4F8A"
CL_COLORS = ["#0E4C92", "#2E9E6B", "#4F86C6", "#8CC7A1", "#0E3462", "#7FB2D9"]
SEQ_RISK = ["#F2F7F4", "#CFE3DA", "#9BC7C8", "#5C93B8", "#1B4F8A"]     # rendah (hijau muda) -> tinggi (biru tua)
SEQ_GOOD = ["#F2F7F4", "#CFE8D9", "#8CC7A1", "#42A06B", "#0E4F33"]
DIVERGE = ["#1B4F8A", "#8FB8DC", "#F2F7F4", "#8CC7A1", "#17784C"]
TH = PALETTE["dark" if st.session_state.dark else "light"]

METHODS = ["K-Means++", "K-Medoids (PAM)", "Agglomerative Ward", "FGWC"]
VARS = ["X1_IR", "X2_CH", "X3_LST", "X4_NDVI", "X5_Kepadatan", "X6_Miskin", "X7_RLS", "X8_Sanitasi"]
LABEL = {"X1_IR": "IR DBD (per 100.000)", "X2_CH": "Curah hujan (mm/tahun)", "X3_LST": "Suhu permukaan/LST (°C)",
         "X4_NDVI": "Kerapatan vegetasi (NDVI)", "X5_Kepadatan": "Kepadatan penduduk (jiwa/km²)",
         "X6_Miskin": "Penduduk miskin (%)", "X7_RLS": "Rata-rata lama sekolah (tahun)", "X8_Sanitasi": "Sanitasi layak (%)"}
SHORT = {"X1_IR": "IR DBD", "X2_CH": "Curah hujan", "X3_LST": "LST", "X4_NDVI": "NDVI",
         "X5_Kepadatan": "Kepadatan", "X6_Miskin": "Kemiskinan", "X7_RLS": "RLS", "X8_Sanitasi": "Sanitasi"}
SUMBER = {"X1_IR": "BPS Jawa Barat 2025 (sumber asli: Dinas Kesehatan)", "X2_CH": "Google Earth Engine — CHIRPS Daily",
          "X3_LST": "Google Earth Engine — MODIS MOD11A2", "X4_NDVI": "Google Earth Engine — Sentinel-2 SR Harmonized",
          "X5_Kepadatan": "BPS Jawa Barat 2025", "X6_Miskin": "BPS Jawa Barat 2025 (Maret 2025)",
          "X7_RLS": "BPS Jawa Barat 2025", "X8_Sanitasi": "BPS Jawa Barat 2025"}
ARTI = {"X1_IR": "Jumlah kasus DBD per 100.000 penduduk; makin tinggi makin berat beban penyakit di wilayah tersebut.",
        "X2_CH": "Total hujan selama setahun; hujan menyediakan genangan tempat nyamuk bertelur.",
        "X3_LST": "Suhu permukaan siang hari dari citra satelit; suhu hangat mempercepat siklus hidup nyamuk dan replikasi virus.",
        "X4_NDVI": "Ukuran kerapatan tumbuhan; nilai rendah menandakan wilayah terbangun atau perkotaan.",
        "X5_Kepadatan": "Jumlah penduduk per km²; makin padat, makin sering kontak manusia dengan nyamuk.",
        "X6_Miskin": "Persentase penduduk miskin; menggambarkan kualitas hunian dan akses layanan kesehatan.",
        "X7_RLS": "Rata-rata lama sekolah; proksi pengetahuan dan perilaku pencegahan (PSN 3M Plus).",
        "X8_Sanitasi": "Persentase rumah tangga dengan sanitasi layak; terkait genangan dan kualitas drainase."}
BAIK_TINGGI = {"X4_NDVI", "X7_RLS", "X8_Sanitasi"}
COLMAP = {"Jumlah Kasus DBD per 100.000 Penduduk": "X1_IR", "X2_Curah_Hujan_mm": "X2_CH", "X3_LST_Celsius": "X3_LST",
          "X4_NDVI": "X4_NDVI", "Kepadatan Penduduk Km2": "X5_Kepadatan", "Persentase Penduduk Miskin": "X6_Miskin",
          "Rata-rata Lama Sekolah": "X7_RLS", "RT Sanitasi Layak (Persen)": "X8_Sanitasi"}

TIM = [("222313120", "Henny Merry Astutik"), ("222313191", "M. Arkillah Ibnu Asshiddqie"),
       ("222313186", "M. Faruq Hafidzullah E."), ("222313205", "Mercy Febriella Liborang"),
       ("222313224", "M. Zidan Kurnia Ahida"), ("222313272", "Nabhan Athallah")]
JUDUL = ("Klasterisasi Risiko Demam Berdarah Dengue Berbasis Data Iklim, Lingkungan, dan Sosial-Ekonomi: "
         "Studi Komparatif Empat Metode Unsupervised Learning di Jawa Barat Tahun 2025")

# --------------------------------------------------------------- ikon SVG (tanpa emoji)
def svg_icon(name, size=20, color=None):
    c = color or GREEN
    P = {
      "mosquito": f'<path d="M12 6c0 3-1.2 5.5-1.2 8.5M12 6c0-2 1.6-3 3-3M12 6c0-2-1.6-3-3-3" stroke="{c}" stroke-width="1.6" fill="none" stroke-linecap="round"/><ellipse cx="12" cy="16" rx="1.5" ry="4.5" fill="{c}"/><path d="M10.5 10c-3-2.5-6-3-8-2 1.8 2.2 4.6 3.6 7.4 4.2M13.5 10c3-2.5 6-3 8-2-1.8 2.2-4.6 3.6-7.4 4.2" fill="none" stroke="{c}" stroke-width="1.4" stroke-linecap="round"/>',
      "map": f'<path d="M9 4 3 6.5v13L9 17l6 2.5 6-2.5v-13L15 6.5 9 4z" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/><path d="M9 4v13M15 6.5v13" stroke="{c}" stroke-width="1.6"/>',
      "chart": f'<path d="M4 20V10M10 20V4M16 20v-7M22 20H2" stroke="{c}" stroke-width="1.8" stroke-linecap="round" fill="none"/>',
      "flask": f'<path d="M9 3h6M10.5 3v6L5 19a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 19l-5.5-10V3" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/><path d="M7.5 15h9" stroke="{c}" stroke-width="1.4"/>',
      "layers": f'<path d="M12 3 3 8l9 5 9-5-9-5z" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/><path d="M3 13l9 5 9-5" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>',
      "award": f'<circle cx="12" cy="9" r="6" fill="none" stroke="{c}" stroke-width="1.6"/><path d="M9 14.5 8 22l4-2 4 2-1-7.5" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>',
      "sliders": f'<path d="M4 6h16M4 12h16M4 18h16" stroke="{c}" stroke-width="1.6" stroke-linecap="round"/><circle cx="9" cy="6" r="2.2" fill="{c}"/><circle cx="15" cy="12" r="2.2" fill="{c}"/><circle cx="7" cy="18" r="2.2" fill="{c}"/>',
      "pin": f'<path d="M12 22s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12z" fill="none" stroke="{c}" stroke-width="1.6"/><circle cx="12" cy="10" r="2.6" fill="{c}"/>',
      "book": f'<path d="M4 4h6a3 3 0 0 1 3 3v13a2.5 2.5 0 0 0-2.5-2H4V4z" fill="none" stroke="{c}" stroke-width="1.6"/><path d="M20 4h-6a3 3 0 0 0-3 3v13a2.5 2.5 0 0 1 2.5-2H20V4z" fill="none" stroke="{c}" stroke-width="1.6"/>',
      "download": f'<path d="M12 3v12m0 0 4-4m-4 4-4-4M4 19h16" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
      "users": f'<circle cx="9" cy="8" r="3.2" fill="none" stroke="{c}" stroke-width="1.6"/><path d="M3 20c0-3.3 2.7-5.5 6-5.5s6 2.2 6 5.5" fill="none" stroke="{c}" stroke-width="1.6"/><path d="M16 5.2A3.2 3.2 0 0 1 16 11M17 14.8c2.4.6 4 2.5 4 5.2" fill="none" stroke="{c}" stroke-width="1.6"/>',
      "home": f'<path d="M3 10.5 12 3l9 7.5V20a1.5 1.5 0 0 1-1.5 1.5h-4V14h-7v7.5h-4A1.5 1.5 0 0 1 3 20v-9.5z" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>',
      "drop": f'<path d="M12 3s6 6.2 6 10a6 6 0 0 1-12 0c0-3.8 6-10 6-10z" fill="none" stroke="{c}" stroke-width="1.6"/>',
      "shield": f'<path d="M12 3 20 6v6c0 5-3.4 8.2-8 9.5C7.4 20.2 4 17 4 12V6l8-3z" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="vertical-align:-3px">{P.get(name, P["drop"])}</svg>'

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_stis.svg")
LOGO_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_stis.png")
def logo_html(h=46):
    import base64
    if os.path.exists(LOGO_PNG):
        b = base64.b64encode(open(LOGO_PNG, "rb").read()).decode()
        return f'<img src="data:image/png;base64,{b}" height="{h}" style="border-radius:8px;background:#fff;padding:3px"/>'
    if os.path.exists(LOGO_PATH):
        b = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
        return f'<img src="data:image/svg+xml;base64,{b}" height="{h}" style="border-radius:8px;background:#fff;padding:3px"/>'
    return ""

pio.templates["dbd"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TH["INK"]),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=TH["PLOT_BG"],
    colorway=CL_COLORS, title=dict(font=dict(size=15, color=TH["INK"])),
    xaxis=dict(gridcolor=TH["LINE"], zerolinecolor=TH["LINE"]),
    yaxis=dict(gridcolor=TH["LINE"], zerolinecolor=TH["LINE"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=TH["LINE"], borderwidth=1),
    margin=dict(l=10, r=10, t=50, b=10)))
pio.templates.default = "dbd"

dark_extra = ("" if not st.session_state.dark else f"""
.stApp, .main {{background:{TH['BG']};}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] input, .stNumberInput input, .stTextInput input {{
  background:{TH['SURF']} !important; color:{TH['INK']} !important; border-color:{TH['LINE']} !important;}}
div[data-baseweb="popover"] div, ul[role="listbox"], li[role="option"] {{background:{TH['SURF']} !important; color:{TH['INK']} !important;}}
.stTabs [data-baseweb="tab"] {{color:{TH['MUTED']};}}
.stTabs [aria-selected="true"] {{background:{TH['SOFT']}; color:{TH['INK']};}}
div[data-testid="stExpander"] details {{background:{TH['SURF']}; border:1px solid {TH['LINE']}; border-radius:10px;}}
button[kind], .stDownloadButton button {{background:{TH['SURF']} !important; color:{TH['INK']} !important; border:1px solid {TH['LINE']} !important;}}
.dbdtab {{width:100%; border-collapse:collapse; font-size:.85rem; color:{TH['INK']};}}
.dbdtab th {{background:{TH['SOFT']}; color:{TH['INK']}; text-align:left; padding:7px 9px; border-bottom:1px solid {TH['LINE']}; position:sticky; top:0;}}
.dbdtab td {{padding:6px 9px; border-bottom:1px solid {TH['LINE']};}}
.dbdtab tr:hover td {{background:{TH['SOFT']};}}
.tabwrap {{max-height:520px; overflow:auto; border:1px solid {TH['LINE']}; border-radius:10px;}}
""")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{font-family:'Inter', 'Segoe UI', sans-serif;}}
.stApp {{background:{TH['BG']};}}
.block-container {{padding-top:1rem; padding-bottom:2.5rem; max-width:1520px;}}
h1,h2,h3,h4,h5 {{color:{TH['INK']}; letter-spacing:-.015em; font-weight:650;}}
p, li, label, span, .stMarkdown, .stCaption {{color:{TH['INK']};}}
div[data-testid="stCaptionContainer"] p {{color:{TH['MUTED']};}}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{background:linear-gradient(175deg,{GREEN_D} 0%,{BLUE_D} 100%); border-right:1px solid {TH['LINE']};}}
section[data-testid="stSidebar"] * {{color:#E9F3EE !important;}}
section[data-testid="stSidebar"] .brand {{display:flex; align-items:center; gap:10px; padding:4px 2px 12px 2px;}}
section[data-testid="stSidebar"] .brand h3 {{color:#FFFFFF !important; margin:0; font-size:1.02rem; line-height:1.25;}}
section[data-testid="stSidebar"] .brand small {{color:#B9D8C8 !important; font-size:.72rem;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
  border-radius:10px; padding:7px 10px; margin-bottom:2px; transition:background .15s ease;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{background:rgba(255,255,255,.10);}}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  background:rgba(255,255,255,.17); box-shadow:inset 3px 0 0 #8CC7A1;}}
section[data-testid="stSidebar"] hr {{border-color:rgba(255,255,255,.18);}}
.sb-fact {{background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); border-radius:10px;
  padding:9px 11px; margin-bottom:7px;}}
.sb-fact b {{display:block; font-size:.93rem;}} .sb-fact span {{font-size:.74rem; color:#BCD8C9 !important;}}

/* ---------- komponen ---------- */
div[data-testid="stMetric"] {{background:{TH['SURF']}; border:1px solid {TH['LINE']}; border-left:4px solid {GREEN};
  border-radius:12px; padding:12px 16px;}}
div[data-testid="stMetricValue"] {{color:{GREEN if not st.session_state.dark else '#8CC7A1'}; font-size:1.5rem;}}
div[data-testid="stMetricLabel"] p {{color:{TH['MUTED']}; font-size:.8rem;}}
.hero {{background:linear-gradient(115deg,{GREEN_D} 0%,{GREEN} 45%,{BLUE} 100%); color:#fff;
  padding:22px 26px; border-radius:16px; margin-bottom:16px; display:flex; align-items:center; gap:18px;}}
.hero h1 {{color:#fff !important; margin:0 0 4px 0; font-size:1.6rem; font-weight:700;}}
.hero p {{color:#E4F1EA !important; margin:0; font-size:.95rem;}}
.hero .ico {{background:rgba(255,255,255,.14); border-radius:12px; padding:10px; line-height:0;}}
.tag {{display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
  padding:2px 11px; border-radius:999px; font-size:.75rem; margin:6px 5px 0 0; color:#fff !important;}}
.card {{background:{TH['SURF']}; border:1px solid {TH['LINE']}; border-radius:13px; padding:15px 17px; height:100%; margin-bottom:10px;}}
.card h4 {{margin:0 0 5px 0; font-size:.97rem; color:{GREEN if not st.session_state.dark else '#8CC7A1'};}}
.card p {{margin:0; color:{TH['MUTED']}; font-size:.87rem;}}
.insight {{background:{TH['SOFT']}; border-left:4px solid {GREEN}; border-radius:10px; padding:12px 15px; margin:9px 0; font-size:.92rem;}}
.warnbox {{background:{TH['SOFT']}; border-left:4px solid {WARN}; border-radius:10px; padding:12px 15px; margin:9px 0; font-size:.92rem;}}
.bluebox {{background:{TH['SOFT']}; border-left:4px solid {BLUE}; border-radius:10px; padding:12px 15px; margin:9px 0; font-size:.92rem;}}
.badge {{display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78rem; font-weight:600;}}
.person {{background:{TH['SURF']}; border:1px solid {TH['LINE']}; border-radius:12px; padding:12px 14px; margin-bottom:9px;
  display:flex; align-items:center; gap:12px;}}
.person .av {{width:38px; height:38px; border-radius:50%; background:linear-gradient(135deg,{GREEN},{BLUE});
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.9rem;}}
.person b {{display:block; font-size:.93rem;}} .person span {{color:{TH['MUTED']}; font-size:.78rem;}}
.footer {{color:{TH['MUTED']}; font-size:.78rem; border-top:1px solid {TH['LINE']}; margin-top:26px; padding-top:10px;
  display:flex; justify-content:space-between; align-items:center; gap:10px;}}
div[data-testid="stDataFrame"] {{border:1px solid {TH['LINE']}; border-radius:10px;}}
.stTabs [data-baseweb="tab-list"] {{gap:4px;}}
.stTabs [data-baseweb="tab"] {{border-radius:9px 9px 0 0; padding:6px 14px;}}
{dark_extra}
</style>""", unsafe_allow_html=True)

def show(fig, **kw):
    try: st.plotly_chart(fig, width="stretch", **kw)
    except TypeError: st.plotly_chart(fig, use_container_width=True, **kw)

def table(df, **kw):
    """Tabel: memakai st.dataframe pada mode terang; tabel HTML bergaya pada mode gelap."""
    if st.session_state.dark:
        try:
            if hasattr(df, "to_html") and not isinstance(df, pd.DataFrame):      # pandas Styler
                html = df.set_table_attributes('class="dbdtab"').to_html()
            else:
                d = df.copy()
                if kw.get("hide_index"): d = d.reset_index(drop=True)
                fmt = {c: "{:,.4g}" for c in d.columns if pd.api.types.is_numeric_dtype(d[c])}
                html = d.style.format(fmt, na_rep="-").hide(axis="index").set_table_attributes('class="dbdtab"').to_html()
            st.markdown(f"<div class='tabwrap'>{html}</div>", unsafe_allow_html=True); return
        except Exception:
            pass
    try: st.dataframe(df, width="stretch", **kw)
    except TypeError: st.dataframe(df, use_container_width=True, **kw)

def hero(title, subtitle, tags=(), icon="drop"):
    t = "".join(f"<span class='tag'>{x}</span>" for x in tags)
    st.markdown(f"""<div class='hero'>
      <div class='ico'>{svg_icon(icon, 30, '#FFFFFF')}</div>
      <div style='flex:1'><h1>{title}</h1><p>{subtitle}</p><div>{t}</div></div>
      <div style='text-align:right'>{logo_html(52)}</div></div>""", unsafe_allow_html=True)

def insight(html): st.markdown(f"<div class='insight'>{html}</div>", unsafe_allow_html=True)
def bluebox(html): st.markdown(f"<div class='bluebox'>{html}</div>", unsafe_allow_html=True)
def warnbox(html): st.markdown(f"<div class='warnbox'>{html}</div>", unsafe_allow_html=True)
def cards(items, ncol=3):
    cols = st.columns(ncol)
    for i, (h, b) in enumerate(items):
        cols[i % ncol].markdown(f"<div class='card'><h4>{h}</h4><p>{b}</p></div>", unsafe_allow_html=True)

# =====================================================================================
# DATA
# =====================================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DBD_DATA_DIR", os.path.join(BASE, "data"))

def find(name):
    for root in [DATA_DIR, os.path.join(DATA_DIR, "output_analisis"), BASE]:
        p = os.path.join(root, name)
        if os.path.exists(p): return p
    return None

@st.cache_data(show_spinner=False)
def load_master(path):
    raw = pd.read_excel(path, sheet_name="Data")
    d = raw.rename(columns=COLMAP).rename(columns={"Kabupaten/Kota": "wilayah"})
    d["kode_wil"] = d["kode_wil"].astype(int)
    d["jenis"] = np.where(d.kode_wil >= 3271, "Kota", "Kabupaten")
    d["wilayah_lengkap"] = np.where(d.jenis == "Kota", d.wilayah, "Kab. " + d.wilayah.astype(str))
    return d.sort_values("kode_wil").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_results(path):
    x = pd.ExcelFile(path)
    return {s: pd.read_excel(x, s) for s in x.sheet_names}

@st.cache_data(show_spinner=False)
def load_geo(geo_path, shp_path):
    if geo_path:
        gj = json.load(open(geo_path, encoding="utf-8"))
    elif shp_path:
        import geopandas as gpd
        g = gpd.read_file(shp_path).to_crs(4326)
        g["kode_wil"] = (pd.to_numeric(g["kdprov"]) * 100 + pd.to_numeric(g["kdkab"])).astype(int)
        gj = json.loads(g[["kode_wil", "geometry"]].to_json())
    else: return None
    for ft in gj["features"]: ft["properties"]["kode_wil"] = int(ft["properties"]["kode_wil"])
    return gj

MASTER, RESULT = find("Master_Dataset_DBD_Jabar_2025.xlsx"), find("Hasil_Klasterisasi_DBD_Jabar_2025.xlsx")
GEO, SHP = find("klaster_jabar_2025.geojson"), find("FIX_KABKOTA_JABAR.shp")
if MASTER is None:
    st.error(f"`Master_Dataset_DBD_Jabar_2025.xlsx` tidak ditemukan di folder `{DATA_DIR}`.")
    st.stop()
df = load_master(MASTER)
R = load_results(RESULT) if RESULT else {}
gj = load_geo(GEO, SHP)
n = len(df)

def sheet(name): return R.get(name, pd.DataFrame()).copy()
def idx_sheet(name):
    d = sheet(name)
    if len(d) and str(d.columns[0]).startswith("Unnamed"):
        d = d.set_index(d.columns[0]); d.index.name = None
    return d

RING = sheet("Ringkasan").iloc[0].to_dict() if len(sheet("Ringkasan")) else {}
LAB = sheet("Label_Klaster")
K_OFF = int(RING.get("K", 2)) if RING else 2
BEST = str(RING.get("BEST", "Agglomerative Ward"))
SELECTED = [v for v in str(RING.get("SELECTED", "X1_IR,X2_CH,X3_LST,X5_Kepadatan,X6_Miskin,X8_Sanitasi")).split(",") if v in VARS]
DROPPED = [v for v in str(RING.get("DROPPED", "")).split(",") if v in VARS]
FG_PAR = dict(m=float(RING.get("FGWC_m", 1.5)), alpha=float(RING.get("FGWC_alpha", .9)),
              a=int(RING.get("FGWC_a", 2)), b=int(RING.get("FGWC_b", 1)))
if len(LAB):
    key = LAB.set_index("kode_wil")
    for m in METHODS:
        if f"klaster_{m}" in key: df[m] = df.kode_wil.map(key[f"klaster_{m}"]).astype(int)
    if "lisa_IR" in key: df["LISA"] = df.kode_wil.map(key["lisa_IR"])
HAS_R = bool(R) and BEST in df.columns

NAMA = {1: "Klaster 1 — Urban padat, kerentanan tinggi", 2: "Klaster 2 — Rural/peri-urban, kerentanan lebih rendah"}
def nama_klaster(c, k=2):
    c = int(c)
    return NAMA.get(c, f"Klaster {c}") if k == 2 else f"Klaster {c}"
def cl_color(c): return CL_COLORS[(int(c) - 1) % len(CL_COLORS)]

REKOM = {1: ["Intensifkan PSN 3M Plus dan Gerakan 1 Rumah 1 Jumantik di permukiman padat.",
             "Surveilans kasus dan jentik mingguan; fogging hanya pada kasus terkonfirmasi.",
             "Kelola kontainer air dan sampah perkotaan; tambah ruang hijau untuk menurunkan suhu kawasan.",
             "Koordinasi kota dengan kabupaten induk karena mobilitas komuter tinggi."],
         2: ["Edukasi PSN berbasis puskesmas dan desa; kewaspadaan dini menjelang musim hujan.",
             "Perbaikan sanitasi dan drainase permukiman.",
             "Pantau wilayah transisi dan kabupaten dengan IR tinggi meski berprofil rural."]}

# =====================================================================================
# FUNGSI ANALITIK (identik dengan notebook) — untuk peta multi-K dan Learning Lab
# =====================================================================================
def transform(d, cols):
    X = d[cols].astype(float).copy()
    if "X1_IR" in X: X["X1_IR"] = np.log1p(X["X1_IR"])
    if "X5_Kepadatan" in X: X["X5_Kepadatan"] = np.log(X["X5_Kepadatan"])
    return X
def zscore(X): return ((X - X.mean()) / X.std(ddof=1)).values
def relabel(labels, ir):
    labs = np.asarray(labels)
    order = sorted(np.unique(labs), key=lambda c: -ir[labs == c].mean())
    mp = {c: i + 1 for i, c in enumerate(order)}
    return np.array([mp[c] for c in labs])
def pam(X, k, metric="cityblock"):
    D = squareform(pdist(X, metric=metric)); nn = len(D); med = [int(np.argmin(D.sum(1)))]
    while len(med) < k:
        cur = D[:, med].min(1)
        med.append(int(np.argmax([(-1 if c in med else np.maximum(cur - D[:, c], 0).sum()) for c in range(nn)])))
    cost = D[:, med].min(1).sum(); improved = True
    while improved:
        improved = False; best = (cost, None)
        for mi in range(k):
            for o in range(nn):
                if o in med: continue
                t = med.copy(); t[mi] = o; c = D[:, t].min(1).sum()
                if c < best[0] - 1e-12: best = (c, (mi, o))
        if best[1]: med[best[1][0]] = best[1][1]; cost = best[0]; improved = True
    return np.argmin(D[:, med], 1), med
def haversine(lon, lat, R_=6371.0):
    lon, lat = np.radians(lon), np.radians(lat)
    a = np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    return 2 * R_ * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
def fgwc(X, k, pop, dist, m=1.5, alpha=.9, a=2, b=1, n_init=30, seed=2025, eps=1e-5, max_iter=500):
    p = np.asarray(pop, float); p = p / p.max()
    with np.errstate(divide="ignore"): Wg = np.outer(p, p) ** b / np.where(dist > 0, dist, np.inf) ** a
    np.fill_diagonal(Wg, 0); rs = Wg.sum(1, keepdims=True); rng = np.random.default_rng(seed); best = None
    for _ in range(n_init):
        U = rng.dirichlet(np.ones(k), size=len(X))
        for _it in range(max_iter):
            Um = U ** m; V = (Um.T @ X) / Um.sum(0)[:, None]
            d = np.maximum(cdist(X, V), 1e-12)
            Uf = 1 / ((d[:, :, None] / d[:, None, :]) ** (2 / (m - 1))).sum(2)
            Un = alpha * Uf + (1 - alpha) * (Wg @ Uf) / rs
            conv = np.abs(Un - U).max() < eps; U = Un
            if conv: break
        Um = U ** m; V = (Um.T @ X) / Um.sum(0)[:, None]; J = (Um * cdist(X, V) ** 2).sum()
        if best is None or J < best[1]: best = (U, J)
    return best[0]

@st.cache_data(show_spinner=False)
def area_km2():
    """Luas poligon (km²) dari GeoJSON dengan proyeksi silinder setara luas (tanpa geopandas)."""
    if gj is None: return None
    R_ = 6371.0; out = {}
    for ft in gj["features"]:
        g = ft["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        tot = 0.0
        for poly in polys:
            for ring_i, ring in enumerate(poly):
                c = np.array(ring, float)
                x = np.radians(c[:, 0]) * R_; y = np.sin(np.radians(c[:, 1])) * R_
                a = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                tot += a if ring_i == 0 else -a
        out[int(ft["properties"]["kode_wil"])] = abs(tot)
    return out

def penduduk():
    ar = area_km2()
    if ar: return (df.X5_Kepadatan.values * df.kode_wil.map(ar).fillna(df.kode_wil.map(ar).median()).values)
    return df.X5_Kepadatan.values.astype(float)

@st.cache_data(show_spinner="Menghitung klaster…")
def cluster_all(k, cols_tuple, fg_par_tuple):
    cols = list(cols_tuple); m_, al, a_, b_ = fg_par_tuple
    Z = zscore(transform(df, cols)); ir = df.X1_IR.values
    res = {}
    res["K-Means++"] = dict(labels=relabel(KMeans(k, init="k-means++", n_init=100, max_iter=300, tol=1e-4,
                                                  random_state=2025).fit_predict(Z), ir))
    lb, med = pam(Z, k); res["K-Medoids (PAM)"] = dict(labels=relabel(lb, ir), medoids=med)
    res["Agglomerative Ward"] = dict(labels=relabel(fcluster(linkage(Z, "ward"), k, "maxclust") - 1, ir))
    U = fgwc(Z, k, penduduk(), haversine(df.lon_c.values, df.lat_c.values), m=m_, alpha=al, a=a_, b=b_)
    lbf = U.argmax(1); new = relabel(lbf, ir); Uo = np.zeros_like(U)
    for o, nw in zip(lbf, new): Uo[:, nw - 1] = U[:, o]
    res["FGWC"] = dict(labels=new, U=Uo)
    for m in METHODS:
        L = res[m]["labels"]
        res[m]["metrik"] = dict(Silhouette=silhouette_score(Z, L) if len(set(L)) > 1 else np.nan,
                                DBI=davies_bouldin_score(Z, L) if len(set(L)) > 1 else np.nan,
                                CH=calinski_harabasz_score(Z, L) if len(set(L)) > 1 else np.nan)
    return res

# =====================================================================================
# PETA
# =====================================================================================
def base_map(fig, height=470, title=""):
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_traces(marker_line_color="white", marker_line_width=0.8, selector=dict(type="choropleth"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=46 if title else 6, b=0), title=title,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.06, x=0), coloraxis_colorbar=dict(thickness=12, len=.7))
    return fig

def add_names(fig, mask=None):
    d = df if mask is None else df[mask]
    fig.add_trace(go.Scattergeo(lon=d.lon_c, lat=d.lat_c, text=d.wilayah, mode="text",
                                textfont=dict(size=8.5, color="#20301F"), hoverinfo="skip", showlegend=False))
    return fig

def hoverframe(extra=None):
    h = pd.DataFrame({"kode_wil": df.kode_wil, "Wilayah": df.wilayah_lengkap, "Jenis": df.jenis,
                      "IR DBD": df.X1_IR, "Kepadatan": df.X5_Kepadatan, "LST": df.X3_LST})
    if extra is not None:
        for k_, v_ in extra.items(): h[k_] = v_
    return h

def map_variable(v, title=None, height=470, names=False, scale=None):
    d = hoverframe({LABEL[v]: df[v]})
    if gj is None:
        fig = px.scatter(d.assign(lon=df.lon_c, lat=df.lat_c), x="lon", y="lat", color=LABEL[v],
                         hover_name="Wilayah", color_continuous_scale=scale or (SEQ_GOOD if v in BAIK_TINGGI else SEQ_RISK))
        fig.update_traces(marker_size=18); return fig
    fig = px.choropleth(d, geojson=gj, locations="kode_wil", featureidkey="properties.kode_wil",
                        color=LABEL[v], hover_name="Wilayah",
                        hover_data={"kode_wil": False, "Jenis": True, "IR DBD": ":.0f"},
                        color_continuous_scale=scale or (SEQ_GOOD if v in BAIK_TINGGI else SEQ_RISK))
    base_map(fig, height, title or LABEL[v])
    if names: add_names(fig)
    return fig

def map_cluster(labels, k=2, title="", height=470, names=False, showlegend=True, extra_hover=None):
    lab = np.asarray(labels).astype(int)
    d = hoverframe(extra_hover); d["Klaster"] = [f"K{c}" for c in lab]
    d["Tipologi"] = [nama_klaster(c, k) for c in lab]
    cmap = {f"K{c}": cl_color(c) for c in np.unique(lab)}
    if gj is None:
        fig = px.scatter(d.assign(lon=df.lon_c, lat=df.lat_c), x="lon", y="lat", color="Klaster",
                         hover_name="Wilayah", color_discrete_map=cmap)
        fig.update_traces(marker_size=18); return fig
    fig = px.choropleth(d, geojson=gj, locations="kode_wil", featureidkey="properties.kode_wil", color="Klaster",
                        hover_name="Wilayah", color_discrete_map=cmap,
                        hover_data={"kode_wil": False, "Tipologi": True, "IR DBD": ":.0f", "Kepadatan": ":,.0f"},
                        category_orders={"Klaster": [f"K{c}" for c in sorted(np.unique(lab))]})
    base_map(fig, height, title); fig.update_layout(showlegend=showlegend)
    if names: add_names(fig)
    return fig

def ringkas_klaster(labels, k=2):
    lab = np.asarray(labels).astype(int); rows = []
    for c in sorted(np.unique(lab)):
        s = df[lab == c]
        rows.append(dict(Klaster=f"K{c}", Tipologi=nama_klaster(c, k), Jumlah=len(s),
                         IR_rata2=s.X1_IR.mean(), Kepadatan_rata2=s.X5_Kepadatan.mean(),
                         LST_rata2=s.X3_LST.mean(), Wilayah=", ".join(s.wilayah_lengkap)))
    return pd.DataFrame(rows)

# =====================================================================================
# SIDEBAR
# =====================================================================================
NAV = [("Ringkasan Eksekutif", "home"), ("Peta Interaktif", "map"), ("Eksplorasi Variabel", "chart"),
       ("Uji Prasyarat", "flask"), ("Penentuan Jumlah Klaster", "layers"), ("Hasil Klasterisasi", "mosquito"),
       ("Evaluasi & Perbandingan", "award"), ("Skenario & Sensitivitas", "sliders"),
       ("Profil Daerah & Rekomendasi", "pin"), ("Learning Lab", "book"), ("Data & Unduhan", "download"),
       ("Tentang & Tim Penyusun", "users")]
PAGES = [p for p, _ in NAV]

with st.sidebar:
    st.markdown(f"""<div class='brand'>{svg_icon('mosquito', 34, '#FFFFFF')}
      <div><h3>Risiko DBD Jawa Barat</h3><small>Tahun referensi 2025 &middot; Kelompok 5</small></div></div>""",
                unsafe_allow_html=True)
    page = st.radio("Navigasi", PAGES, label_visibility="collapsed")
    st.divider()
    if HAS_R:
        st.markdown(f"<div class='sb-fact'><b>{BEST}</b><span>metode terpilih (TOPSIS)</span></div>"
                    f"<div class='sb-fact'><b>K = {K_OFF}</b><span>jumlah klaster optimal</span></div>"
                    f"<div class='sb-fact'><b>{len(SELECTED)} dari 8 variabel</b><span>input klaster; NDVI &amp; RLS jadi variabel profil</span></div>",
                    unsafe_allow_html=True)
    else:
        st.warning("Berkas hasil notebook belum ditemukan. Salin Hasil_Klasterisasi_DBD_Jabar_2025.xlsx "
                   "dan klaster_jabar_2025.geojson ke folder data/.")
    if gj is None: st.info("GeoJSON/SHP tidak ditemukan - peta ditampilkan sebagai titik.")
    st.divider()
    _dark = st.toggle("Mode gelap", value=st.session_state.dark, key="dark_toggle") if hasattr(st, "toggle") \
        else st.checkbox("Mode gelap", value=st.session_state.dark, key="dark_toggle")
    if _dark != st.session_state.dark:
        st.session_state.dark = _dark
        st.rerun()
    st.caption("Sumber data: BPS Provinsi Jawa Barat 2025; Google Earth Engine (CHIRPS, MODIS, Sentinel-2); shapefile BPS.")

def footer():
    st.markdown(f"""<div class='footer'><span>Proyek Data Mining Kelompok 5 &middot; Program Studi Komputasi Statistik (D-IV),
      Peminatan Sains Data &middot; Politeknik Statistika STIS</span><span>{logo_html(26)}</span></div>""", unsafe_allow_html=True)

# =====================================================================================
# 1. RINGKASAN EKSEKUTIF
# =====================================================================================
if page == "Ringkasan Eksekutif":
    hero("Klasterisasi Risiko Demam Berdarah Dengue — Jawa Barat 2025",
         "Pengelompokan 27 kabupaten/kota berdasarkan profil iklim, lingkungan, dan sosial-ekonomi, "
         "dengan membandingkan empat metode unsupervised learning.",
         ["27 kabupaten/kota", "8 variabel", "4 metode", f"K = {K_OFF}", f"Metode terpilih: {BEST}"], icon="mosquito")
    ev = idx_sheet("Evaluasi")
    c = st.columns(5)
    c[0].metric("Rata-rata IR DBD", f"{df.X1_IR.mean():.1f}", "kasus per 100.000 penduduk", delta_color="off")
    top = df.loc[df.X1_IR.idxmax()]
    c[1].metric("IR tertinggi", f"{top.X1_IR:.0f}", top.wilayah_lengkap, delta_color="off")
    low = df.loc[df.X1_IR.idxmin()]
    c[2].metric("IR terendah", f"{low.X1_IR:.0f}", low.wilayah_lengkap, delta_color="off")
    if HAS_R:
        n1 = int((df[BEST] == 1).sum())
        c[3].metric("Wilayah kerentanan tinggi", f"{n1}", f"dari {n} wilayah", delta_color="off")
        if len(ev) and BEST in ev.index:
            c[4].metric("Kualitas & stabilitas", f"{ev.loc[BEST,'Silhouette']:.3f} | {ev.loc[BEST,'Jaccard']:.2f}",
                        "Silhouette | Jaccard", delta_color="off")
    if HAS_R:
        left, right = st.columns([1.35, 1])
        with left:
            show(map_cluster(df[BEST], K_OFF, f"Tipologi kerentanan DBD — {BEST}", names=True))
            st.caption("Warna merah = klaster kerentanan tinggi. Arahkan kursor ke sebuah wilayah untuk melihat IR DBD dan kepadatan penduduknya.")
        with right:
            st.markdown("#### Empat temuan utama")
            hp = f"{RING['H']:.3f}" if "H" in RING else "-"
            pmc = f"{RING.get('p_MC', float('nan')):.3f}" if "p_MC" in RING else "-"
            mi, mp = RING.get("Moran_I_IR", float('nan')), RING.get("Moran_p", float('nan'))
            insight(f"<b>1. Data layak diklasterkan.</b> Statistik Hopkins H = {hp} dengan p Monte Carlo = {pmc}, "
                    "artinya pengelompokan wilayah bukan kebetulan.")
            insight(f"<b>2. Risiko DBD tidak menular ke wilayah tetangga.</b> Moran's I = {mi:.3f} (p = {mp:.3f}). "
                    "Kota dengan IR tinggi justru dikelilingi kabupaten ber-IR rendah, sehingga yang menentukan adalah karakter kota–desa, bukan kedekatan lokasi.")
            insight("<b>3. Terbentuk dua tipologi.</b> Wilayah urban padat berkerentanan tinggi dan wilayah rural/peri-urban "
                    "berkerentanan lebih rendah; pembedanya suhu permukaan, kerapatan vegetasi, kepadatan penduduk, dan kemiskinan.")
            insight(f"<b>4. Hasil konsisten antarmetode.</b> K-Means++, Ward, dan FGWC memberi pembagian yang sama persis; "
                    f"TOPSIS memilih {BEST} sebagai metode terbaik.")
        st.markdown("#### Ringkasan kedua klaster")
        rk = ringkas_klaster(df[BEST], K_OFF)
        for _, r in rk.iterrows():
            st.markdown(f"<span class='badge' style='background:{cl_color(r.Klaster[1])}22;color:{cl_color(r.Klaster[1])}'>{r.Tipologi}</span> "
                        f"&nbsp;<b>{r.Jumlah} wilayah</b> · rata-rata IR {r.IR_rata2:.1f} · kepadatan {r.Kepadatan_rata2:,.0f} jiwa/km² · LST {r.LST_rata2:.1f} °C", unsafe_allow_html=True)
            st.caption(r.Wilayah)
        with st.expander("Apa arti dashboard ini bagi pengambil kebijakan?"):
            st.markdown("""
- **Tipologi**, bukan peringkat. Wilayah dikelompokkan berdasarkan kemiripan profil risiko, sehingga intervensi dapat disamakan dalam satu kelompok.
- **Klaster 1** menuntut penguatan PSN, surveilans rutin, dan pengelolaan lingkungan perkotaan.
- **Klaster 2** menuntut perbaikan sanitasi/drainase dan kewaspadaan dini musiman.
- Hasil ini **bersifat asosiatif**, bukan sebab-akibat, dan berlaku untuk kondisi tahun 2025.
""")
    footer()

# =====================================================================================
# 2. PETA INTERAKTIF
# =====================================================================================
elif page == "Peta Interaktif":
    hero("Peta Interaktif", "Bandingkan sebaran variabel, hasil klaster antarmetode, dan pengaruh jumlah klaster (K).",
         ["Pilih variabel", "Pilih metode", "Pilih K", "Bandingkan berdampingan"], icon="map")
    mode = st.radio("Tampilkan peta", ["Sebaran variabel", "Hasil klaster (satu metode)", "Bandingkan antarmetode"], horizontal=True)
    names = st.toggle("Tampilkan nama wilayah", value=True) if hasattr(st, "toggle") else st.checkbox("Tampilkan nama wilayah", True)

    if mode == "Sebaran variabel":
        c1, c2 = st.columns([1, 1])
        v = c1.selectbox("Variabel", VARS, format_func=lambda k: LABEL[k])
        v2 = c2.selectbox("Bandingkan dengan (opsional)", ["— tidak ada —"] + VARS,
                          format_func=lambda k: k if k.startswith("—") else LABEL[k])
        cols = st.columns(2 if v2 != "— tidak ada —" else 1)
        with cols[0]: show(map_variable(v, LABEL[v], names=names))
        if v2 != "— tidak ada —":
            with cols[1]: show(map_variable(v2, LABEL[v2], names=names))
        hi = df.nlargest(3, v)[["wilayah_lengkap", v]]; lo = df.nsmallest(3, v)[["wilayah_lengkap", v]]
        insight(f"<b>{LABEL[v]}.</b> {ARTI[v]}<br>Tertinggi: " +
                ", ".join(f"{r.wilayah_lengkap} ({r[v]:,.2f})" for _, r in hi.iterrows()) +
                "<br>Terendah: " + ", ".join(f"{r.wilayah_lengkap} ({r[v]:,.2f})" for _, r in lo.iterrows()) +
                f"<br>Korelasi dengan IR DBD: r = {df[v].corr(df.X1_IR):+.3f} (Spearman ρ = {df[v].corr(df.X1_IR, method='spearman'):+.3f}). "
                f"Sumber data: {SUMBER[v]}.")

    elif mode == "Hasil klaster (satu metode)":
        c1, c2, c3 = st.columns([1.2, 1, 1])
        mt = c1.selectbox("Metode", METHODS, index=METHODS.index(BEST) if BEST in METHODS else 0)
        k = c2.slider("Jumlah klaster (K)", 2, 6, K_OFF)
        pakai_resmi = (k == K_OFF) and HAS_R
        res = cluster_all(k, tuple(SELECTED), tuple(FG_PAR.values()))
        lab = df[mt].values if pakai_resmi and mt in df else res[mt]["labels"]
        c3.metric("Silhouette", f"{res[mt]['metrik']['Silhouette']:.3f}")
        left, right = st.columns([1.4, 1])
        with left:
            show(map_cluster(lab, k, f"{mt} — K = {k}", names=names, height=520))
        with right:
            st.markdown("**Ringkasan klaster**")
            rk = ringkas_klaster(lab, k)
            for _, r in rk.iterrows():
                st.markdown(f"<span class='badge' style='background:{cl_color(r.Klaster[1])}22;color:{cl_color(r.Klaster[1])}'>{r.Klaster}</span> "
                            f"**{r.Jumlah} wilayah** · IR rata-rata {r.IR_rata2:.1f} · kepadatan {r.Kepadatan_rata2:,.0f}", unsafe_allow_html=True)
                st.caption(r.Wilayah)
            if mt == "FGWC" and "U" in res["FGWC"]:
                U = res["FGWC"]["U"]; tr = df.wilayah_lengkap.values[U.max(1) < .6]
                st.markdown("**Wilayah transisi FGWC** (keanggotaan maksimum < 0,60): " + (", ".join(tr) if len(tr) else "tidak ada"))
        if pakai_resmi:
            st.success(f"Peta ini adalah hasil resmi laporan (K = {K_OFF}).")
        else:
            ari = adjusted_rand_score(df[BEST], lab) if HAS_R else np.nan
            warnbox(f"K = {k} merupakan <b>eksplorasi tambahan</b>, bukan hasil resmi laporan (K optimal = {K_OFF}). "
                    f"Kemiripan dengan hasil resmi: ARI = {ari:.3f}. Silhouette pada K ini = {res[mt]['metrik']['Silhouette']:.3f}.")

    else:
        c1, c2 = st.columns([1, 2])
        k = c1.slider("Jumlah klaster (K)", 2, 6, K_OFF, key="kcmp")
        pilih = c2.multiselect("Metode yang dibandingkan", METHODS, default=METHODS)
        if not pilih: st.stop()
        res = cluster_all(k, tuple(SELECTED), tuple(FG_PAR.values()))
        labs = {m: (df[m].values if (k == K_OFF and HAS_R and m in df) else res[m]["labels"]) for m in pilih}
        cols = st.columns(2)
        for i, m in enumerate(pilih):
            with cols[i % 2]:
                show(map_cluster(labs[m], k, f"{m} · Silhouette {res[m]['metrik']['Silhouette']:.3f}",
                                 height=380, names=False, showlegend=(i == 0)), key=f"cmp{m}{k}")
        mt_tab = pd.DataFrame({m: res[m]["metrik"] for m in pilih}).T.round(3)
        mt_tab.insert(0, "Ukuran klaster", [", ".join(str(int(x)) for x in np.bincount(labs[m])[1:]) for m in pilih])
        st.markdown("**Perbandingan kualitas partisi pada K = %d**" % k); table(mt_tab)
        if len(pilih) > 1:
            A = pd.DataFrame(index=pilih, columns=pilih, dtype=float)
            for a_, b_ in itertools.product(pilih, pilih): A.loc[a_, b_] = adjusted_rand_score(labs[a_], labs[b_])
            show(px.imshow(A.astype(float), text_auto=".2f", color_continuous_scale=SEQ_GOOD, zmin=0, zmax=1,
                           title="Kemiripan hasil antarmetode (Adjusted Rand Index; 1 = identik)", height=360))
            same = [f"{a_} = {b_}" for a_, b_ in itertools.combinations(pilih, 2) if A.loc[a_, b_] == 1]
            insight("<b>Cara membaca.</b> ARI = 1 berarti kedua metode membagi wilayah dengan cara yang persis sama; "
                    "ARI mendekati 0 berarti kesepakatannya setara tebakan acak. " +
                    (f"Pada K = {k}, metode dengan hasil identik: {', '.join(same)}." if same else f"Pada K = {k} tidak ada pasangan metode yang identik."))
    footer()

# =====================================================================================
# 3. EKSPLORASI VARIABEL
# =====================================================================================
elif page == "Eksplorasi Variabel":
    hero("Eksplorasi Variabel", "Statistik ringkas, sebaran, dan hubungan antarvariabel penyusun tipologi risiko.", icon="chart")
    v = st.selectbox("Pilih variabel", VARS, format_func=lambda k: LABEL[k])
    x = df[v].astype(float)
    st.caption(f"**{LABEL[v]}** — {ARTI[v]} Sumber: {SUMBER[v]}. Peran: " +
               ("variabel input klaster." if v in SELECTED else "variabel profil (tidak dipakai sebagai input karena multikolinearitas)."))
    c = st.columns(5)
    c[0].metric("Rata-rata", f"{x.mean():,.2f}"); c[1].metric("Median", f"{x.median():,.2f}")
    c[2].metric("Minimum", f"{x.min():,.2f}"); c[3].metric("Maksimum", f"{x.max():,.2f}")
    g1 = ((x - x.mean()) ** 3).mean() / x.std() ** 3
    c[4].metric("Kemencengan (g₁)", f"{g1:.2f}", "menceng kuat" if abs(g1) > 1 else "relatif simetris", delta_color="off")
    col1, col2 = st.columns([1.15, 1])
    with col1: show(map_variable(v, f"Sebaran {LABEL[v]}", names=True))
    with col2:
        d = df.sort_values(v)
        fig = px.bar(d, x=v, y="wilayah_lengkap", orientation="h", color="jenis", height=500,
                     color_discrete_map={"Kota": RISK, "Kabupaten": GREEN}, labels={v: LABEL[v], "wilayah_lengkap": "", "jenis": ""})
        fig.add_vline(x=x.mean(), line_dash="dash", line_color=TH["MUTED"], annotation_text="rata-rata provinsi")
        show(fig)
    col3, col4 = st.columns(2)
    with col3:
        show(px.histogram(df, x=v, nbins=10, marginal="box", color_discrete_sequence=[GREEN],
                          labels={v: LABEL[v]}, title="Distribusi dan pencilan"))
    with col4:
        U, pmw = stats.mannwhitneyu(df.loc[df.jenis == "Kota", v], df.loc[df.jenis == "Kabupaten", v])
        fig = px.box(df, x="jenis", y=v, points="all", color="jenis", hover_name="wilayah_lengkap",
                     color_discrete_map={"Kota": RISK, "Kabupaten": GREEN}, labels={v: LABEL[v], "jenis": ""},
                     title=f"Kota vs kabupaten (Mann-Whitney p = {pmw:.4f})")
        show(fig)
    st.markdown("#### Hubungan antarvariabel")
    c1, c2 = st.columns(2)
    with c1:
        meth = st.radio("Jenis korelasi", ["pearson", "spearman"], horizontal=True)
        Rm = df[VARS].corr(method=meth).rename(index=SHORT, columns=SHORT)
        show(px.imshow(Rm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=500,
                       title=f"Matriks korelasi {meth.capitalize()}"))
    with c2:
        xa = st.selectbox("Sumbu X", VARS, index=4, format_func=lambda k: LABEL[k])
        ya = st.selectbox("Sumbu Y", VARS, index=0, format_func=lambda k: LABEL[k])
        color = df[BEST].astype(str).map(lambda s: f"K{s}") if HAS_R else df.jenis
        fig = px.scatter(df, x=xa, y=ya, color=color, hover_name="wilayah_lengkap", height=430,
                         color_discrete_map={f"K{i+1}": CL_COLORS[i] for i in range(6)} | {"Kota": RISK, "Kabupaten": GREEN},
                         labels={xa: LABEL[xa], ya: LABEL[ya], "color": "Klaster"})
        r_, p_ = stats.spearmanr(df[xa], df[ya])
        fig.update_traces(marker_size=13); fig.update_layout(title=f"ρ Spearman = {r_:+.3f} (p = {p_:.4f})")
        show(fig)
    insight("<b>Cara membaca korelasi.</b> Nilai mendekati +1 atau −1 berarti dua variabel bergerak seiring atau berlawanan. "
            "Dengan 27 wilayah, |r| di atas sekitar 0,38 sudah signifikan pada taraf 5%. Korelasi bukan bukti sebab-akibat.")
    footer()

# =====================================================================================
# 4. UJI PRASYARAT
# =====================================================================================
elif page == "Uji Prasyarat":
    hero("Uji Prasyarat Analisis Klaster", "Pemeriksaan sebelum pemodelan: redundansi variabel, kelayakan pengelompokan, dan pola spasial.", icon="flask")
    t1, t2, t3 = st.tabs(["Multikolinearitas (VIF)", "Kecenderungan klaster (Hopkins)", "Pola spasial (Moran's I & LISA)"])
    with t1:
        if "VIF" in R:
            vf = idx_sheet("VIF"); col = [c for c in vf.columns if "akhir" in str(c)]
            col = col[0] if col else vf.columns[-1]
            d = vf.reset_index().rename(columns={"index": "Kode"})
            fig = px.bar(d, x=col, y="Label", orientation="h", height=400,
                         color=(d[col] > 10).map({True: "VIF > 10 (redundan)", False: "VIF ≤ 10 (aman)"}),
                         color_discrete_map={"VIF > 10 (redundan)": RISK, "VIF ≤ 10 (aman)": GREEN},
                         labels={col: "VIF pada bentuk akhir variabel", "Label": "", "color": ""})
            fig.add_vline(x=10, line_dash="dash", line_color=TH["MUTED"]); show(fig)
            table(vf)
        insight("<b>Apa itu VIF?</b> Ukuran seberapa besar informasi sebuah variabel sudah terkandung di variabel lain. "
                f"VIF > 10 berarti redundan dan membuat satu dimensi terhitung ganda pada jarak antarwilayah. "
                f"Karena itu <b>{', '.join(LABEL[v] for v in DROPPED) or '—'}</b> dikeluarkan dari input, tetapi tetap dipakai untuk menjelaskan profil klaster. "
                "VIF dihitung pada bentuk akhir variabel (setelah transformasi logaritma), sesuai bentuk yang dipakai model.")
    with t2:
        if RING:
            c = st.columns(3)
            c[0].metric("Hopkins H (utama)", f"{RING['H']:.3f}")
            c[1].metric("p-value Monte Carlo", f"{RING.get('p_MC', float('nan')):.4f}",
                        "Tolak H₀: ada struktur klaster" if RING.get("p_MC", 1) < .05 else "Gagal tolak H₀", delta_color="off")
            c[2].metric("H_d (pelengkap)", f"{RING['H_d']:.3f}")
            if "Hopkins_MC_null" in R:
                nl = sheet("Hopkins_MC_null")
                fig = px.histogram(nl, x="H_null", nbins=35, color_discrete_sequence=["#B9C8BC"], height=380,
                                   labels={"H_null": "Nilai H bila data benar-benar acak"},
                                   title="Uji Monte Carlo: posisi H data kita dibanding 1.000 data acak")
                fig.add_vline(x=float(np.quantile(nl.H_null, .99)), line_dash="dash", line_color=TH["MUTED"],
                              annotation_text="batas 99% data acak")
                fig.add_vline(x=RING["H"], line_color=RISK, line_width=3, annotation_text=f"H data kita = {RING['H']:.3f}")
                show(fig)
            if "Hopkins_MC" in R: table(idx_sheet("Hopkins_MC").round(4))
            insight("<b>Cara membaca.</b> Jika data tidak berpola, nilai H berada di sekitar 0,5. Nilai H data ini berada jauh di kanan "
                    "seluruh simulasi data acak, sehingga pengelompokan wilayah memang bermakna. Karena H masih di bawah 0,70, "
                    "kekuatan pengelompokan tergolong <b>moderat</b>: batas antarklaster ada, tetapi tidak setajam data yang terpisah sempurna.")
        if "Mahalanobis" in R:
            mh = sheet("Mahalanobis")
            fig = px.bar(mh, x="Wilayah", y="D2", height=380, color="Pencilan_chi2",
                         color_discrete_map={True: RISK, False: "#9FB3A4"}, title="Pemeriksaan pencilan multivariat (jarak Mahalanobis)")
            fig.add_hline(y=26.12, line_dash="dash", line_color=TH["MUTED"], annotation_text="ambang χ²(0,999; 8) = 26,12"); show(fig)
            st.caption("Tidak ada wilayah yang melewati ambang, sehingga tidak ada pencilan multivariat; seluruh wilayah dipertahankan.")
    with t3:
        if "Moran_I" in R: table(sheet("Moran_I").round(4), hide_index=True)
        if "Moran_Sensitivitas" in R:
            st.markdown("**Pemeriksaan ulang dengan definisi tetangga dan transformasi lain**")
            table(sheet("Moran_Sensitivitas").round(4), hide_index=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if "LISA" in df:
                lmap = {"High-High": RISK, "Low-Low": BLUE, "High-Low": "#F4A9A0", "Low-High": "#A8C8E8", "Tidak signifikan": "#D9E0DA"}
                d = hoverframe({"LISA": df.LISA})
                if gj is not None:
                    fig = px.choropleth(d, geojson=gj, locations="kode_wil", featureidkey="properties.kode_wil", color="LISA",
                                        hover_name="Wilayah", color_discrete_map=lmap, hover_data={"kode_wil": False, "IR DBD": ":.0f"})
                    base_map(fig, 430, "Peta LISA IR DBD (α = 0,05)"); show(fig)
        with cc2:
            ls = sheet("LISA_IR")
            if len(ls):
                fig = px.scatter(ls, x="z", y="lag_z", hover_name="Wilayah", color="Kuadran", height=430,
                                 title="Moran scatterplot: IR wilayah vs rata-rata IR tetangganya")
                b = np.polyfit(ls.z, ls.lag_z, 1); xs = np.linspace(ls.z.min(), ls.z.max(), 10)
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", name=f"kemiringan = {b[0]:.3f}",
                                         line=dict(dash="dash", color=TH["MUTED"])))
                fig.add_hline(y=0, line_color=TH["LINE"]); fig.add_vline(x=0, line_color=TH["LINE"])
                fig.update_traces(marker_size=11, selector=dict(mode="markers")); show(fig)
        insight("<b>Mengapa Moran's I negatif dan tidak signifikan?</b> Sebagian besar kota di Jawa Barat merupakan enklave di dalam kabupaten induknya, "
                "misalnya Kota Cirebon (IR 374) di tengah Kabupaten Cirebon (IR 48). Wilayah bertetangga justru berbeda tajam, sehingga beban DBD "
                "lebih ditentukan oleh karakter kota–desa daripada kedekatan geografis. Determinannya (curah hujan, suhu, kemiskinan, sanitasi) tetap "
                "mengelompok secara spasial, sehingga metode berbobot geografis (FGWC) tetap layak diuji.")
    footer()

# =====================================================================================
# 5. PENENTUAN K
# =====================================================================================
elif page == "Penentuan Jumlah Klaster":
    hero("Penentuan Jumlah Klaster Optimal", "Berapa kelompok yang paling sesuai untuk 27 wilayah? Keputusan diambil dari banyak indeks sekaligus.", icon="layers")
    if "Penentuan_K" in R:
        pk = sheet("Penentuan_K")
        ind = st.radio("Indeks", ["Silhouette", "DBI", "CH", "Dunn"], horizontal=True)
        arah = "makin tinggi makin baik" if ind != "DBI" else "makin rendah makin baik"
        fig = px.line(pk, x="K", y=ind, color="Metode", markers=True, height=430, title=f"{ind} pada berbagai K ({arah})")
        if ind == "Silhouette":
            fig.add_hline(y=.5, line_dash="dash", line_color=TH["MUTED"], annotation_text="target laporan 0,50")
            fig.add_hline(y=.25, line_dash="dot", line_color=TH["MUTED"], annotation_text="batas minimal 0,25")
        fig.add_vrect(x0=K_OFF - .18, x1=K_OFF + .18, fillcolor=GREEN, opacity=.12, line_width=0)
        fig.update_xaxes(dtick=1); show(fig)
        c1, c2 = st.columns([1, 1])
        with c1:
            vt = sheet("Voting_K"); vt.columns = ["Indeks", "K terpilih"]
            cnt = vt["K terpilih"].value_counts().reindex(range(2, 7), fill_value=0).reset_index()
            cnt.columns = ["K", "Jumlah suara"]
            show(px.bar(cnt, x="K", y="Jumlah suara", text="Jumlah suara", height=380,
                        color_discrete_sequence=[GREEN], title="Voting 17 indeks validitas"))
        with c2: table(vt, hide_index=True, height=380)
        insight(f"<b>Kesimpulan.</b> Mayoritas indeks memilih <b>K = {K_OFF}</b>, dan seluruh metode menghasilkan klaster dengan anggota memadai. "
                "Dua klaster juga mudah diterjemahkan menjadi kebijakan: wilayah kerentanan tinggi dan lebih rendah. "
                "Nilai silhouette yang berada di bawah 0,50 menunjukkan batas antarkelompok tidak tajam, karena karakter wilayah berubah bertahap dari rural ke urban.")
    footer()

# =====================================================================================
# 6. HASIL KLASTERISASI
# =====================================================================================
elif page == "Hasil Klasterisasi":
    hero("Hasil Klasterisasi Resmi", f"Hasil laporan: K = {K_OFF}, empat metode dibandingkan pada data yang sama.", icon="mosquito")
    if not HAS_R: st.stop()
    m = st.selectbox("Metode", METHODS, index=METHODS.index(BEST))
    lab = df[m].values
    c1, c2 = st.columns([1.3, 1])
    with c1: show(map_cluster(lab, K_OFF, f"{m}", names=True, height=520))
    with c2:
        X = df[VARS]; mm = (X - X.min()) / (X.max() - X.min())
        prof = mm.groupby(lab).mean()
        fig = go.Figure()
        for c_, r_ in prof.iterrows():
            fig.add_trace(go.Scatterpolar(r=list(r_) + [r_.iloc[0]], theta=[SHORT[v] for v in VARS] + [SHORT[VARS[0]]],
                                          fill="toself", name=f"K{c_}", line_color=cl_color(c_)))
        fig.update_layout(height=520, polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                          title="Profil klaster (skala 0–1)", legend=dict(orientation="h"))
        show(fig)
    pr = df.groupby(m)[VARS].mean().rename(columns=SHORT)
    pr.insert(0, "Jumlah wilayah", df[m].value_counts().sort_index())
    pr.index = [nama_klaster(i, K_OFF) for i in pr.index]
    st.markdown("**Rata-rata setiap variabel per klaster (skala asli)**"); table(pr.round(2))
    if m == "FGWC" and "U_FGWC" in R:
        U = sheet("U_FGWC"); uk = [c for c in U.columns if str(c).startswith("K")]
        U["Keanggotaan maks"] = U[uk].max(axis=1); U["Status"] = np.where(U["Keanggotaan maks"] < .6, "Transisi", "Tegas")
        a, b = st.columns(2)
        with a:
            vals = df.wilayah_lengkap.map(U.set_index("Wilayah")[uk[0]]).astype(float)
            d = hoverframe({"Keanggotaan K1": vals})
            if gj is not None:
                fig = px.choropleth(d, geojson=gj, locations="kode_wil", featureidkey="properties.kode_wil",
                                    color="Keanggotaan K1", hover_name="Wilayah", color_continuous_scale=SEQ_RISK,
                                    hover_data={"kode_wil": False, "IR DBD": ":.0f"})
                base_map(fig, 430, "Derajat keanggotaan terhadap Klaster 1"); show(fig)
        with b:
            fig = px.bar(U.sort_values("Keanggotaan maks"), x="Keanggotaan maks", y="Wilayah", color="Status",
                         orientation="h", height=430, color_discrete_map={"Transisi": WARN, "Tegas": GREEN},
                         title="Seberapa yakin sebuah wilayah masuk satu klaster")
            fig.add_vline(x=.6, line_dash="dash", line_color=TH["MUTED"]); show(fig)
        insight("<b>Keunggulan FGWC.</b> Setiap wilayah memperoleh derajat keanggotaan, bukan sekadar label. Wilayah dengan keanggotaan "
                "maksimum di bawah 0,60 berada di antara dua tipologi sehingga perlu kewaspadaan khusus.")
    if m == "K-Medoids (PAM)":
        res = cluster_all(K_OFF, tuple(SELECTED), tuple(FG_PAR.values()))
        st.success("Wilayah representatif (medoid): " + ", ".join(df.wilayah_lengkap.values[res[m]["medoids"]]))
    st.markdown("**Kesepakatan dengan metode lain**")
    ar = {mm_: adjusted_rand_score(df[m], df[mm_]) for mm_ in METHODS if mm_ in df}
    st.write(" · ".join(f"{k_}: ARI {v_:.3f}" for k_, v_ in ar.items() if k_ != m))
    footer()

# =====================================================================================
# 7. EVALUASI
# =====================================================================================
elif page == "Evaluasi & Perbandingan":
    hero("Evaluasi & Perbandingan Metode", "Lima dimensi penilaian dirangkum menjadi satu peringkat dengan metode TOPSIS.", icon="award")
    if "Evaluasi" not in R: st.stop()
    ev = idx_sheet("Evaluasi")
    ren = {"Silhouette": "Silhouette ↑", "DBI": "DBI ↓", "CH": "CH ↑", "Dunn": "Dunn ↑", "Jaccard": "Stabilitas ↑",
           "RK": "Koherensi spasial ↑", "Eps2_IR": "Efek IR (ε²) ↑", "Waktu_s": "Waktu (detik) ↓"}
    tampil = ev[[c for c in ren if c in ev.columns]].rename(columns=ren)
    st.markdown("**Metrik evaluasi setiap metode**"); table(tampil.style.format(precision=4))
    cards([("Validitas internal", "Seberapa rapat anggota satu klaster dan seberapa terpisah antarklaster (Silhouette, DBI, CH, Dunn)."),
           ("Stabilitas", "Apakah klaster tetap terbentuk ketika sebagian wilayah dikeluarkan secara acak (Jaccard bootstrap)."),
           ("Koherensi spasial", "Seberapa sering wilayah bertetangga berada di klaster yang sama."),
           ("Kebermaknaan", "Apakah IR DBD benar-benar berbeda antarklaster (uji Kruskal-Wallis dan ukuran efek ε²)."),
           ("Efisiensi", "Waktu komputasi; seluruh metode jauh di bawah batas 10 detik."),
           ("TOPSIS", "Menggabungkan seluruh kriteria menjadi satu skor 0–1; makin dekat ke 1 makin baik.")], 3)
    st.markdown("#### Peringkat TOPSIS — geser bobot untuk menguji sendiri")
    crit = {"Silhouette": ("benefit", .15), "DBI": ("cost", .10), "CH": ("benefit", .05), "Dunn": ("benefit", .05),
            "Jaccard": ("benefit", .20), "RK": ("benefit", .15), "Eps2_IR": ("benefit", .20), "Waktu_s": ("cost", .10)}
    crit = {k_: v_ for k_, v_ in crit.items() if k_ in ev.columns}
    cols = st.columns(len(crit)); w = {}
    for col, (k_, (kind, w0)) in zip(cols, crit.items()):
        w[k_] = col.slider(ren.get(k_, k_).replace(" ↑", "").replace(" ↓", ""), 0.0, .5, w0, .05)
    ws = np.array(list(w.values()))
    if ws.sum() == 0: st.warning("Semua bobot bernilai nol."); st.stop()
    X = ev[list(crit)].astype(float).values; Rn = X / np.sqrt((X ** 2).sum(0)); V = Rn * (ws / ws.sum())
    kind = np.array([v_[0] for v_ in crit.values()])
    Ap = np.where(kind == "benefit", V.max(0), V.min(0)); An = np.where(kind == "benefit", V.min(0), V.max(0))
    Dp = np.sqrt(((V - Ap) ** 2).sum(1)); Dn = np.sqrt(((V - An) ** 2).sum(1))
    tp = pd.DataFrame({"Skor TOPSIS": Dn / (Dp + Dn)}, index=ev.index).sort_values("Skor TOPSIS", ascending=False)
    tp["Peringkat"] = range(1, len(tp) + 1)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.bar(tp.reset_index(), x="Skor TOPSIS", y="index", orientation="h", height=330,
                     text=tp["Skor TOPSIS"].round(3).values, color="Skor TOPSIS", color_continuous_scale=SEQ_GOOD,
                     labels={"index": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False); show(fig)
    with c2:
        table(tp.round(4)); st.success(f"Metode terbaik dengan bobot saat ini: **{tp.index[0]}**")
        st.caption("Bobot otomatis dinormalkan agar berjumlah 1. Bobot awal sesuai Tabel 3.6 laporan. "
                   "Coba geser bobot *Waktu* menjadi 0 untuk melihat pengaruh kriteria waktu.")
    c3, c4 = st.columns(2)
    with c3:
        if "ARI" in R:
            A = idx_sheet("ARI").astype(float)
            show(px.imshow(A, text_auto=".2f", color_continuous_scale=SEQ_GOOD, zmin=0, zmax=1, height=400,
                           title="Kemiripan hasil antarmetode (ARI)"))
    with c4:
        if "Stabilitas_Jaccard" in R:
            jb = sheet("Stabilitas_Jaccard")
            fig = px.bar(jb, x="Metode", y="Jaccard_rata2", color=jb.Klaster.astype(str), barmode="group", height=400,
                         color_discrete_map={"1": CL_COLORS[0], "2": CL_COLORS[1]},
                         labels={"Jaccard_rata2": "Jaccard", "color": "Klaster"}, title="Stabilitas klaster (bootstrap)")
            fig.add_hline(y=.75, line_dash="dash", line_color=TH["MUTED"], annotation_text="batas stabil 0,75"); show(fig)
    with st.expander("Kruskal-Wallis: variabel apa yang paling membedakan klaster?"):
        if "Kruskal_Wallis" in R:
            kw = sheet("Kruskal_Wallis")
            pv = kw.pivot(index="Variabel", columns="Metode", values="eps2")
            show(px.imshow(pv, text_auto=".2f", color_continuous_scale=SEQ_GOOD, height=420,
                           title="Ukuran efek ε² (0,26 ke atas = efek besar)"))
    if "Kriteria_Sukses" in R:
        ks = sheet("Kriteria_Sukses").copy()
        ks["Status"] = np.where(ks["Tercapai"], "Tercapai", "Belum tercapai")
        st.markdown("#### Capaian kriteria kesuksesan (Tabel 1.1 laporan)")
        table(ks.drop(columns="Tercapai"), hide_index=True)
        warnbox("<b>Dua kriteria belum tercapai</b>, yaitu signifikansi Moran's I dan silhouette ≥ 0,50. Keduanya merupakan "
                "karakteristik data (pola kota enklave dan gradasi rural–urban), bukan kegagalan metode, dan dilaporkan sebagai keterbatasan.")
    footer()

# =====================================================================================
# 8. SKENARIO
# =====================================================================================
elif page == "Skenario & Sensitivitas":
    hero("Skenario Validasi & Sensitivitas", "Apakah tipologi tetap sama bila variabel, transformasi, atau jumlah klaster diubah?", icon="sliders")
    t1, t2, t3 = st.tabs(["Skenario B — tanpa IR", "Skenario C — sensitivitas", "Skenario D — PCA"])
    with t1:
        if "Skenario_B" in R:
            table(idx_sheet("Skenario_B").round(4))
            insight("<b>Validasi eksternal.</b> Klaster dibentuk hanya dari faktor penyebab (tanpa memasukkan angka kesakitan). "
                    "IR DBD tetap berbeda signifikan antarklaster, sehingga tipologi yang terbentuk memang berkaitan dengan beban penyakit.")
    with t2:
        if "Skenario_C" in R:
            c = sheet("Skenario_C")
            show(px.imshow(c.pivot(index="Varian", columns="Metode", values="ARI_vs_A"), text_auto=".2f",
                           color_continuous_scale=SEQ_GOOD, zmin=0, zmax=1, height=400,
                           title="Kemiripan hasil dengan skenario utama (ARI)"))
            insight("ARI ≥ 0,65 menandakan hasil kokoh. Perubahan besar hanya terjadi ketika jumlah klaster dipaksa menjadi 3, "
                    "yang sejalan dengan turunnya nilai silhouette pada K = 3.")
    with t3:
        if "Skenario_D_PCA" in R:
            table(idx_sheet("Skenario_D_PCA").round(4))
            if "PCA_Loading" in R:
                show(px.imshow(idx_sheet("PCA_Loading").astype(float), text_auto=".2f", color_continuous_scale="RdBu_r",
                               zmin=-1, zmax=1, height=420, title="Loading komponen utama"))
            if RING:
                st.caption(f"KMO = {RING.get('KMO', float('nan')):.3f} (≥ 0,5 layak) · Bartlett p = {RING.get('Bartlett_p', float('nan')):.3g}")
            insight("<b>Alternatif tanpa membuang variabel.</b> Seluruh delapan variabel diringkas menjadi komponen utama. "
                    "Hasil klasternya identik dengan skenario utama, sehingga keputusan mengeluarkan NDVI dan RLS tidak mengubah tipologi.")
    if "FGWC_alpha" in R:
        st.markdown("#### Pengaruh bobot geografis pada FGWC")
        fa = sheet("FGWC_alpha")
        show(px.line(fa.melt(id_vars=["alpha"], value_vars=[c for c in ["MPC", "Silhouette", "RK", "ARI_vs_terpilih"] if c in fa]),
                     x="alpha", y="value", color="variable", markers=True, height=380,
                     labels={"alpha": "α (makin kecil = pengaruh geografis makin besar)", "value": "nilai", "variable": ""}))
    footer()

# =====================================================================================
# 9. PROFIL DAERAH
# =====================================================================================
elif page == "Profil Daerah & Rekomendasi":
    hero("Profil Kabupaten/Kota", "Bandingkan sebuah wilayah dengan rata-rata provinsi dan lihat rekomendasi intervensinya.", icon="pin")
    w_ = st.selectbox("Pilih kabupaten/kota", df.wilayah_lengkap)
    r = df[df.wilayah_lengkap == w_].iloc[0]
    cl = int(r[BEST]) if HAS_R else 1
    c = st.columns(4)
    c[0].metric("IR DBD", f"{r.X1_IR:.0f}", f"{(r.X1_IR / df.X1_IR.mean() - 1) * 100:+.0f}% vs rata-rata provinsi")
    c[1].metric("Peringkat IR", f"{int(df.X1_IR.rank(ascending=False)[r.name])} dari 27")
    c[2].metric("Kepadatan", f"{r.X5_Kepadatan:,.0f}", "jiwa/km²", delta_color="off")
    if "U_FGWC" in R:
        U = sheet("U_FGWC").set_index("Wilayah"); uk = [c_ for c_ in U.columns if str(c_).startswith("K")]
        um = float(U.loc[w_, uk].max())
        c[3].metric("Keyakinan klaster (FGWC)", f"{um:.2f}", "wilayah transisi" if um < .6 else "tegas", delta_color="off")
    col1, col2 = st.columns([1.25, 1])
    with col1:
        rel = pd.DataFrame({"Variabel": [SHORT[v] for v in VARS],
                            "Selisih": [(r[v] / df[v].mean() - 1) * 100 for v in VARS],
                            "Nilai": [r[v] for v in VARS]})
        fig = px.bar(rel, x="Selisih", y="Variabel", orientation="h", height=430, custom_data=["Nilai"],
                     color=(rel.Selisih > 0).map({True: "Di atas rata-rata", False: "Di bawah rata-rata"}),
                     color_discrete_map={"Di atas rata-rata": RISK, "Di bawah rata-rata": BLUE},
                     labels={"Selisih": "Selisih terhadap rata-rata provinsi (%)", "Variabel": "", "color": ""},
                     title=f"Profil {w_}")
        fig.update_traces(hovertemplate="%{y}: %{customdata[0]:,.2f} (%{x:+.1f}%)")
        show(fig)
    with col2:
        st.markdown(f"<span class='badge' style='background:{cl_color(cl)}22;color:{cl_color(cl)}'>{nama_klaster(cl, K_OFF)}</span>", unsafe_allow_html=True)
        if HAS_R:
            st.markdown("**Klaster menurut tiap metode:** " + " · ".join(f"{m}: K{int(r[m])}" for m in METHODS if m in df))
        st.markdown("**Rekomendasi intervensi**")
        for t in REKOM.get(cl, REKOM[2]): st.markdown(f"- {t}")
        if HAS_R:
            st.caption("Wilayah lain dalam klaster yang sama: " +
                       ", ".join(df.loc[(df[BEST] == cl) & (df.wilayah_lengkap != w_), "wilayah_lengkap"]))
    if HAS_R:
        st.markdown("#### Posisi wilayah ini pada peta")
        show(map_cluster(df[BEST], K_OFF, "", height=380, names=True,
                         extra_hover={"Dipilih": np.where(df.wilayah_lengkap == w_, "wilayah dipilih", "")}))
    st.markdown("#### Matriks kebijakan")
    mat = pd.DataFrame({"Aspek intervensi": ["PSN 3M Plus & Jumantik", "Surveilans & kewaspadaan dini", "Sanitasi & drainase",
                                             "Edukasi masyarakat", "Koordinasi antarwilayah"],
                        "Klaster 1 (urban padat)": ["Prioritas utama", "Mingguan", "Sedang", "Kampanye perkotaan", "Kota–kabupaten induk"],
                        "Klaster 2 (rural/peri-urban)": ["Rutin", "Musiman (musim hujan)", "Prioritas utama", "Berbasis desa & puskesmas", "Wilayah transisi"]})
    table(mat, hide_index=True)
    footer()

# =====================================================================================
# 10. LEARNING LAB
# =====================================================================================
elif page == "Learning Lab":
    hero("Learning Lab", "Memahami metode di balik dashboard: konsep, percobaan langsung, simulasi, dan kuis.",
         ["Untuk masyarakat umum", "Untuk praktisi kesehatan", "Untuk dosen & mahasiswa"], icon="book")
    t1, t2, t3, t4 = st.tabs(["Konsep & rumus", "Coba sendiri", "Simulasi K-Means", "Kuis"])
    with t1:
        with st.expander("Apa itu klasterisasi (clustering)?", expanded=True):
            st.markdown("Klasterisasi mengelompokkan wilayah berdasarkan **kemiripan profil**, tanpa label yang sudah ditentukan sebelumnya. "
                        "Tujuannya: wilayah dalam satu kelompok semirip mungkin, dan antarkelompok seberbeda mungkin.")
        with st.expander("Mengapa data distandardisasi?"):
            st.latex(r"z_{ij}=\frac{x_{ij}-\bar{x}_j}{s_j}")
            st.markdown("Curah hujan bernilai ribuan, NDVI bernilai desimal. Tanpa penyetaraan skala, jarak antarwilayah akan didominasi curah hujan.")
        with st.expander("VIF: menghindari informasi berulang"):
            st.latex(r"VIF_j=\frac{1}{1-R_j^2}")
            st.markdown("VIF > 10 berarti variabel tersebut hampir bisa ditebak dari variabel lain, sehingga satu dimensi terhitung dua kali.")
        with st.expander("Hopkins: apakah data layak diklasterkan?"):
            st.latex(r"H=\frac{\sum u_i}{\sum u_i+\sum w_i}")
            st.markdown("H sekitar 0,5 berarti data acak. Signifikansi diuji dengan membandingkan H terhadap 1.000 data acak buatan (Monte Carlo).")
        with st.expander("Moran's I: apakah wilayah bertetangga mirip?"):
            st.latex(r"I=\frac{n}{S_0}\frac{\sum_i\sum_j w_{ij}(x_i-\bar x)(x_j-\bar x)}{\sum_i(x_i-\bar x)^2}")
            st.markdown("I > 0 berarti tetangga mirip, I < 0 berarti tetangga berbeda (pola papan catur), I ≈ −1/(n−1) berarti acak.")
        with st.expander("Empat metode yang dibandingkan"):
            st.markdown("**K-Means++** — pusat klaster berupa titik rata-rata; cepat dan menjadi pembanding dasar.")
            st.latex(r"J=\sum_k\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2")
            st.markdown("**K-Medoids (PAM)** — pusat klaster berupa wilayah nyata sehingga tahan terhadap nilai ekstrem.")
            st.latex(r"J=\sum_k\sum_{x_i\in C_k} d(x_i,m_k)")
            st.markdown("**Agglomerative Ward** — menggabungkan wilayah bertahap dan menghasilkan dendrogram.")
            st.latex(r"\Delta ESS=\frac{n_An_B}{n_A+n_B}\lVert\mu_A-\mu_B\rVert^2")
            st.markdown("**FGWC** — versi fuzzy yang memperhitungkan jarak dan populasi wilayah lain (Hukum Tobler).")
            st.latex(r"u'_{ik}=\alpha u_{ik}+\beta\frac{\sum_j w_{ij}u_{jk}}{\sum_j w_{ij}},\quad w_{ij}=\frac{(p_ip_j)^b}{d_{ij}^a}")
        with st.expander("Cara menilai hasil klaster"):
            st.latex(r"s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\qquad C_i=\frac{D_i^-}{D_i^++D_i^-}")
            st.markdown("Silhouette 0,51–0,70 = struktur wajar; 0,26–0,50 = lemah. Jaccard ≥ 0,75 = klaster stabil. "
                        "TOPSIS menggabungkan semua kriteria menjadi satu skor.")
    with t2:
        st.markdown("Ubah pilihan di bawah dan lihat langsung pengaruhnya terhadap peta dan kualitas klaster.")
        c1, c2, c3 = st.columns([2, 1, 1])
        vars_ = c1.multiselect("Variabel input", VARS, default=SELECTED, format_func=lambda k: LABEL[k])
        kk = c2.slider("Jumlah klaster", 2, 6, K_OFF)
        mt = c3.selectbox("Metode", METHODS)
        par = dict(FG_PAR)
        if mt == "FGWC":
            e1, e2, e3, e4 = st.columns(4)
            par["m"] = e1.select_slider("m (kekaburan)", [1.5, 2.0, 2.5], FG_PAR["m"])
            par["alpha"] = e2.slider("α (bobot atribut)", .5, .9, FG_PAR["alpha"], .1)
            par["a"] = e3.select_slider("a (pengaruh jarak)", [1, 2], FG_PAR["a"])
            par["b"] = e4.select_slider("b (pengaruh populasi)", [1, 2], FG_PAR["b"])
        if len(vars_) < 2: st.warning("Pilih minimal dua variabel."); st.stop()
        Z = zscore(transform(df, vars_)); t0 = time.perf_counter()
        if mt == "FGWC":
            U = fgwc(Z, kk, penduduk(), haversine(df.lon_c.values, df.lat_c.values), **par)
            lab = relabel(U.argmax(1), df.X1_IR.values)
        elif mt == "K-Means++": lab = relabel(KMeans(kk, n_init=50, random_state=2025).fit_predict(Z), df.X1_IR.values)
        elif mt == "K-Medoids (PAM)": lab = relabel(pam(Z, kk)[0], df.X1_IR.values)
        else: lab = relabel(fcluster(linkage(Z, "ward"), kk, "maxclust") - 1, df.X1_IR.values)
        dt = time.perf_counter() - t0
        vif = np.diag(np.linalg.pinv(np.corrcoef(Z, rowvar=False)))
        cm = st.columns(5)
        cm[0].metric("Silhouette", f"{silhouette_score(Z, lab):.3f}" if len(set(lab)) > 1 else "—")
        cm[1].metric("DBI", f"{davies_bouldin_score(Z, lab):.3f}" if len(set(lab)) > 1 else "—")
        cm[2].metric("CH", f"{calinski_harabasz_score(Z, lab):.1f}" if len(set(lab)) > 1 else "—")
        cm[3].metric("VIF maksimum", f"{vif.max():.1f}", "ada redundansi" if vif.max() > 10 else "aman", delta_color="off")
        cm[4].metric("Waktu hitung", f"{dt*1000:.0f} ms")
        p1, p2 = st.columns([1.2, 1])
        with p1: show(map_cluster(lab, kk, f"{mt} · K = {kk} · {len(vars_)} variabel", names=True, height=470))
        with p2:
            Zc = Z - Z.mean(0); _, _, vt_ = np.linalg.svd(Zc, full_matrices=False); P = Zc @ vt_[:2].T
            fig = px.scatter(x=P[:, 0], y=P[:, 1], color=[f"K{c}" for c in lab], hover_name=df.wilayah_lengkap,
                             color_discrete_map={f"K{i+1}": CL_COLORS[i] for i in range(6)}, height=470,
                             labels={"x": "Komponen 1", "y": "Komponen 2", "color": "Klaster"},
                             title="Peta kemiripan wilayah (proyeksi 2 dimensi)")
            fig.update_traces(marker_size=14); show(fig)
        if HAS_R:
            st.caption(f"Kemiripan dengan hasil resmi laporan ({BEST}, K = {K_OFF}): ARI = {adjusted_rand_score(df[BEST], lab):.3f}")
    with t3:
        st.markdown("K-Means bekerja berulang: **menempelkan wilayah ke pusat terdekat**, lalu **memindahkan pusat** ke rata-rata anggotanya.")
        Z = zscore(transform(df, SELECTED)); Zc = Z - Z.mean(0)
        _, _, vt_ = np.linalg.svd(Zc, full_matrices=False); P = Zc @ vt_[:2].T
        c1, c2 = st.columns(2)
        kk2 = c1.slider("Jumlah klaster", 2, 5, 2, key="ksim")
        seed = c2.number_input("Seed titik awal", 0, 999, 7)
        rng = np.random.default_rng(seed); C = P[rng.choice(len(P), kk2, replace=False)]; hist = []
        for _ in range(12):
            lab = cdist(P, C).argmin(1); hist.append((C.copy(), lab.copy()))
            Cn = np.array([P[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(kk2)])
            if np.allclose(Cn, C): break
            C = Cn
        it = st.slider("Iterasi", 1, len(hist), 1)
        C, lab = hist[it - 1]
        fig = px.scatter(x=P[:, 0], y=P[:, 1], color=[f"K{c+1}" for c in lab], hover_name=df.wilayah_lengkap, height=480,
                         color_discrete_map={f"K{i+1}": CL_COLORS[i] for i in range(6)},
                         labels={"x": "Komponen 1", "y": "Komponen 2", "color": "Klaster"},
                         title=f"Iterasi {it} dari {len(hist)}" + (" (konvergen)" if it == len(hist) else ""))
        fig.update_traces(marker_size=13)
        fig.add_trace(go.Scatter(x=C[:, 0], y=C[:, 1], mode="markers", name="Pusat klaster",
                                 marker=dict(symbol="x", size=18, color="black", line=dict(width=2))))
        show(fig)
        st.caption(f"Total jarak kuadrat ke pusat (WSS) pada iterasi ini: {sum(((P[lab==j]-C[j])**2).sum() for j in range(kk2)):.2f} — nilainya menurun tiap iterasi.")
    with t4:
        Q = [("Mengapa variabel distandardisasi sebelum klasterisasi?",
              ["Agar data menjadi normal", "Agar setiap variabel berkontribusi setara pada jarak", "Agar jumlah klaster bertambah"], 1),
             ("H Hopkins = 0,587 dengan p Monte Carlo = 0,001 berarti …",
              ["Data acak", "Ada kecenderungan klaster yang signifikan dengan kekuatan moderat", "Terjadi multikolinearitas"], 1),
             ("Moran's I IR DBD negatif dan tidak signifikan. Artinya …",
              ["Wilayah bertetangga cenderung mirip", "Wilayah bertetangga cenderung berbeda; pola kota enklave", "Analisis gagal"], 1),
             ("Metode mana yang pusat klasternya berupa wilayah nyata?", ["K-Means++", "K-Medoids (PAM)", "FGWC"], 1),
             ("Stabilitas Jaccard 0,95 menunjukkan …", ["Klaster sangat stabil", "Klaster tidak stabil", "Silhouette tinggi"], 0),
             ("Nilai tambah FGWC dibanding tiga metode lain adalah …",
              ["Paling cepat", "Memberi derajat keanggotaan dan memperhitungkan kedekatan geografis", "Tidak perlu menentukan K"], 1)]
        skor = 0
        for i, (q, opts, ans) in enumerate(Q):
            a = st.radio(f"**{i+1}. {q}**", opts, index=None, key=f"q{i}")
            if a is not None:
                if opts.index(a) == ans: st.success("Benar."); skor += 1
                else: st.error(f"Belum tepat. Jawaban: {opts[ans]}")
        st.metric("Skor", f"{skor} / {len(Q)}")
    footer()

# =====================================================================================
# 11. DATA & UNDUHAN
# =====================================================================================
elif page == "Data & Unduhan":
    hero("Data & Unduhan", "Data mentah, label klaster, dan seluruh hasil analisis.", icon="download")
    kolom = ["kode_wil", "wilayah_lengkap", "jenis"] + VARS + [m for m in METHODS if m in df]
    out = df[kolom].rename(columns={**SHORT, "wilayah_lengkap": "Wilayah", "jenis": "Jenis"})
    table(out, hide_index=True, height=520)
    c1, c2 = st.columns(2)
    c1.download_button("Unduh data + label klaster (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "klaster_dbd_jabar_2025.csv", "text/csv")
    if RESULT:
        with open(RESULT, "rb") as f:
            c2.download_button("Unduh seluruh hasil analisis (Excel)", f.read(), os.path.basename(RESULT))
    st.markdown("#### Keterangan variabel")
    table(pd.DataFrame({"Kode": VARS, "Variabel": [LABEL[v] for v in VARS], "Arti": [ARTI[v] for v in VARS],
                        "Sumber": [SUMBER[v] for v in VARS],
                        "Peran": ["Input klaster" if v in SELECTED else "Variabel profil" for v in VARS]}), hide_index=True)
    footer()

# =====================================================================================
# 12. TENTANG & TIM PENYUSUN
# =====================================================================================
else:
    hero("Tentang Proyek & Tim Penyusun",
         "Identitas laporan, tim, dan catatan teknis dashboard.",
         ["Data Mining", "Semester 6", "Tahun akademik 2025/2026"], icon="users")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("#### Judul laporan")
        st.markdown(f"<div class='card'><p style='font-size:.95rem;color:{TH['INK']}'>{JUDUL}</p></div>", unsafe_allow_html=True)
        st.markdown("#### Tim penyusun — Kelompok 5")
        for i, (nim, nama) in enumerate(TIM, start=1):
            ini = "".join([w[0] for w in nama.replace(".", "").split()[:2]]).upper()
            st.markdown(f"<div class='person'><div class='av'>{ini}</div><div><b>{nama}</b>"
                        f"<span>NIM {nim}</span></div><div style='margin-left:auto;color:{TH['MUTED']};font-size:.8rem'>Anggota {i}</div></div>",
                        unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h4>Program studi</h4><p>Komputasi Statistik, Program Diploma IV &mdash; "
                    f"Peminatan Sains Data, Politeknik Statistika STIS</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("#### Ringkasan proyek")
        cards([("Tujuan", "Menyusun tipologi kerentanan DBD 27 kabupaten/kota Jawa Barat 2025 sebagai dasar prioritas intervensi."),
               ("Kerangka kerja", "CRISP-DM enam fase, dengan tambahan uji prasyarat klaster dan penentuan K."),
               ("Metode", "K-Means++, K-Medoids (PAM), Agglomerative Ward, dan Fuzzy Geographically Weighted Clustering."),
               ("Evaluasi", "Validitas internal, stabilitas bootstrap, koherensi spasial, Kruskal-Wallis, efisiensi, lalu TOPSIS.")], 1)
        st.markdown("#### Teknologi")
        cards([("Analisis", "Python: pandas, NumPy, SciPy, scikit-learn; implementasi PAM, FGWC, Hopkins, VAT/iVAT, Moran's I, dan TOPSIS ditulis langsung dari rumus."),
               ("Data spasial", "Shapefile BPS 27 kabupaten/kota; peta interaktif Plotly."),
               ("Dashboard", "Streamlit; seluruh angka dibaca dari berkas ekspor notebook.")], 1)
    st.markdown("#### Sumber data")
    table(pd.DataFrame({"Kelompok": ["Kesehatan & sosial-ekonomi", "Iklim & lingkungan", "Batas wilayah"],
                        "Rincian": ["IR DBD, kepadatan penduduk, penduduk miskin, rata-rata lama sekolah, sanitasi layak",
                                    "Curah hujan CHIRPS, suhu permukaan MODIS MOD11A2, NDVI Sentinel-2 (cadangan Landsat 8/9)",
                                    "Shapefile administrasi 27 kabupaten/kota"],
                        "Sumber": ["BPS Provinsi Jawa Barat 2025", "Google Earth Engine, periode 1 Januari-31 Desember 2025",
                                   "Badan Pusat Statistik"]}), hide_index=True)
    st.markdown("#### Catatan penggunaan")
    bluebox("Hasil klasterisasi bersifat <b>asosiatif</b>, bukan sebab-akibat, dan berlaku untuk kondisi tahun 2025. "
            "Tipologi sebaiknya diperbarui setiap tahun ketika data BPS dan citra satelit terbaru tersedia.")
    footer()
