# app.py — Quanskill Analytics (pro theme + tabs + NLP streaming + optional AI insights)
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt  # Prophet components view
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import re
import json
import base64
import mimetypes
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- MUST be first Streamlit call ---
page_icon_path = os.getenv("PAGE_ICON_PATH", "logo.png") if os.path.exists(os.getenv("PAGE_ICON_PATH", "logo.png")) else None
st.set_page_config(page_title="Quanskill Analytics", page_icon=page_icon_path, layout="wide")

# ---- Optional libs (Prophet; NumPy 2 patch) --------------------------------
try:
    import numpy as _np

    if not hasattr(_np, "float_"):
        _np.float_ = _np.float64
    if not hasattr(_np, "int_"):
        _np.int_ = _np.int64
except Exception:
    pass

USE_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    USE_PROPHET = False

# ---- Azure OpenAI (v1) ------------------------------------------------------
try:
    from openai import AzureOpenAI

    HAS_AZURE = True
except Exception:
    HAS_AZURE = False

# ---------------- CONFIG -----------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "outputs_v2")
LOGO1_PATH = os.getenv("LOGO1_PATH", "logo.png")  # SHTP or partner
LOGO2_PATH = os.getenv("LOGO2_PATH", "/Saigon_Hi-Tech_Park.png")  # Quanskill
APP_TITLE = os.getenv("APP_TITLE", "Hệ thống phân tích và giám sát năng lượng cho nhà máy xử lý gỗ dăm")

# ---------------- THEME (light, professional) --------------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility:hidden;}
/* unify all top bars with light bg */
[data-testid="stHeader"], [data-testid="stDecoration"] { background: #f6f7fb !important; }
:root{
  --bg:#f6f7fb; --surface:#ffffff; --ink:#0f172a; --muted:#5b6375;
  --border:#e6e8ef; --accent:#ff8a1f; --accent-ink:#1a1a1a;
  --success-bg:#eaf7ee; --success-bd:#cde8d6; --success-txt:#0f5132;
  --dark:#0b0f15; --dark-bd:#1f2937; --dark-ink:#e5e7eb;
}
.stApp { background: var(--bg); }

/* Sidebar (bg light, text forced to dark) */
section[data-testid="stSidebar"]{
  background:#fff;
  border-right:1px solid var(--border);
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .streamlit-expanderHeader{
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea{
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder{
  color: var(--muted) !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span{
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] .stButton>button{
  color: var(--accent-ink) !important;
}

/* ---------- MAIN SCREEN: force all text to dark on light backgrounds ---------- */
[data-testid="stAppViewContainer"]{
  color: var(--ink) !important;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] small,
[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stCaption,
[data-testid="stAppViewContainer"] .streamlit-expanderHeader{
  color: var(--ink) !important;
}
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] select,
[data-testid="stAppViewContainer"] textarea{
  color: var(--ink) !important;
}
[data-testid="stAppViewContainer"] input::placeholder,
[data-testid="stAppViewContainer"] textarea::placeholder{
  color: var(--muted) !important;
}
/* chips/tags in main area */
[data-testid="stAppViewContainer"] [data-baseweb="tag"] span{
  color: var(--ink) !important;
}
/* Plotly text (axis labels, legends) - FORCE WHITE TEXT */
.js-plotly-plot .plotly .main-svg text{
  fill: #ffffff !important;
}
/* Plotly legend text - FORCE WHITE */
.js-plotly-plot .plotly .legend text{
  fill: #ffffff !important;
}
/* Plotly hover labels - FORCE WHITE */
.js-plotly-plot .plotly .hoverlayer text{
  fill: #ffffff !important;
}

/* Brand area */
.brand-card{ padding:14px 10px 10px 10px; border-bottom:1px solid var(--border); }
.brand-logos{ display:flex; align-items:center; gap:10px; justify-content:flex-start; }
.brand-logos img{ height:36px; width:auto; object-fit:contain; display:block; }

/* Headings */
h1,h2,h3{ color:var(--ink); font-weight:800; letter-spacing:.2px; }
.subtle{ color:var(--muted); }

/* Cards & KPI */
.card{ background:var(--surface); border:1px solid var(--border); border-radius:16px;
       box-shadow:0 6px 24px rgba(15,23,42,.06); padding:18px; }
.kpi{ display:flex; flex-direction:column; gap:6px; }
.kpi .label{ color:var(--muted); font-weight:600; font-size:.92rem; }
.kpi .value{ font-weight:800; font-size:1.8rem; color:var(--ink); }

/* Status pill */
.status-ok{ background:var(--success-bg); border:1px solid var(--success-bd);
            color:var(--success-txt); border-radius:12px; padding:10px 14px; display:inline-block; }

/* Tabs (top, pill) */
div[data-baseweb="tab-list"] { gap:8px !important; }
button[role="tab"]{
  background:var(--surface) !important; border:1px solid var(--border) !important;
  color:var(--muted) !important; padding:10px 16px !important; border-radius:16px !important;
  box-shadow:0 4px 14px rgba(15,23,42,.04);
}
button[role="tab"][aria-selected="true"]{
  background: linear-gradient(180deg,#fff,#f9fafc) !important;
  color:var(--ink) !important; border-color:#d9dce4 !important; font-weight:800 !important;
}

/* Buttons */
.stButton>button{
  background:var(--accent); color:var(--accent-ink); border:0; font-weight:700;
  border-radius:12px; padding:.6rem 1rem;
}
.stButton>button:hover{ filter:brightness(.98); }

/* Plotly bg harmonize */
.js-plotly-plot .plotly .bg { background-color: transparent !important; }

/* AI dark boxes (for insights & streaming) — keep these white-on-black */
.ai-box{
  background:var(--dark); color:var(--dark-ink) !important;
  border:1px solid var(--dark-bd); border-radius:14px; padding:14px 16px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  white-space:pre-wrap;
}
.ai-box *{ color:var(--dark-ink) !important; }
.ai-title{ font-weight:800; margin-bottom:6px; }

/* --- Sidebar dark-input text should be WHITE (previous fix) --- */
section[data-testid="stSidebar"] .stDateInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[role="combobox"] input,
section[data-testid="stSidebar"] .stMultiSelect div[role="combobox"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] input {
  color: var(--dark-ink) !important;
}
section[data-testid="stSidebar"] .stDateInput input::placeholder,
section[data-testid="stSidebar"] .stNumberInput input::placeholder,
section[data-testid="stSidebar"] .stTextInput input::placeholder{
  color: #9ca3af !important;
}
section[data-testid="stSidebar"] .stNumberInput button,
section[data-testid="stSidebar"] .stNumberInput button *{
  color: var(--dark-ink) !important;
  fill: var(--dark-ink) !important;
}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"]{
  color: var(--dark-ink) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- HELPERS ----------------------------------------------------


def to_data_uri(path: str):
    if not path or not os.path.exists(path):
        return None
    mt, _ = mimetypes.guess_type(path)
    mt = mt or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mt};base64,{b64}"


@st.cache_data(show_spinner=False)
def load_csv(p):
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_all(data_dir):
    paths = {
        "long": os.path.join(data_dir, "energy_gold_detail_long.csv"),
        "d_dn": os.path.join(data_dir, "energy_gold_daily_by_daynight.csv"),
        "d_tar": os.path.join(data_dir, "energy_gold_daily_by_tariff.csv"),
        "d_eq": os.path.join(data_dir, "energy_gold_daily_equipment.csv"),
        "wide": os.path.join(data_dir, "energy_gold_detail_wide.csv"),
        "d_ca": os.path.join(data_dir, "energy_gold_daily_by_ca.csv"),
    }
    dfs, missing = {}, []
    for k, p in paths.items():
        if os.path.exists(p):
            dfs[k] = load_csv(p)
        else:
            missing.append(p)
    return dfs, missing


def coerce_date(df, col="date"):
    if df is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def kpi(v, label):
    v = float(0 if pd.isna(v) else v)
    st.markdown(
        f"""
    <div class="card kpi">
      <div class="label">{label}</div>
      <div class="value">{v:,.2f}</div>
    </div>""",
        unsafe_allow_html=True,
    )


def daterange(df, col="date"):
    if df is None or df.empty or col not in df.columns:
        return df, None
    dmin, dmax = df[col].min(), df[col].max()
    if pd.isna(dmin) or pd.isna(dmax):
        return df, None
    picked = st.sidebar.date_input("Khoảng thời gian", [dmin.date(), dmax.date()])
    start = pd.to_datetime(picked[0])
    end = pd.to_datetime(picked[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df[col] >= start) & (df[col] <= end)].copy(), (start, end)


def ms_filter(df, col, label):
    if df is None or col not in df.columns:
        return df
    opts = ["(Tất cả)"] + sorted(df[col].dropna().astype(str).unique().tolist())
    sel = st.sidebar.multiselect(label, opts, default="(Tất cả)")
    if "(Tất cả)" in sel or not sel:
        return df
    return df[df[col].astype(str).isin(sel)]


def strong_cap(series):
    s = series.dropna().astype(float)
    if s.empty:
        return None
    q99 = s.quantile(0.99)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    mad = (s - s.median()).abs().median() * 1.4826
    cap = min(q99, s.median() + 4 * iqr, s.median() + 6 * mad)
    return max(cap, s.quantile(0.90))


def cap_for_plot(df, ycol):
    d = df.copy()
    hi = strong_cap(d[ycol])
    clipped = 0
    if hi is not None:
        clipped = int((d[ycol] > hi).sum())
        d[ycol] = np.minimum(d[ycol], hi)
    return d, clipped, hi


# ---------- JSON sanitiser (fixes Timestamp serialization) ----------


def json_ready(obj):
    """Recursively convert Pandas/NumPy/datetime objects to JSON-safe Python values."""
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso"))
    if isinstance(obj, pd.Series):
        return json_ready(obj.to_frame())
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(obj).isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    return obj


# ---------- Forecast helpers ----------


def train_forecast(
    df,
    date_col="date",
    y_col="energy_kwh",
    horizon=60,
    use_prophet=True,
    weekly=True,
    yearly=True,
    fourier_weekly=6,
    fourier_yearly=10,
    changepoint_prior=0.05,
    seasonality_prior=10.0,
):
    s = df.groupby(date_col)[y_col].sum().reset_index().sort_values(date_col).dropna()
    if s.empty or len(s) < 10:
        raise ValueError("Không đủ dữ liệu để mô hình hóa.")
    if use_prophet and USE_PROPHET:
        d = s.rename(columns={date_col: "ds", y_col: "y"})
        m = Prophet(
            changepoint_prior_scale=changepoint_prior,
            seasonality_prior_scale=seasonality_prior,
        )
        if weekly:
            m.add_seasonality(name="weekly", period=7, fourier_order=fourier_weekly)
        if yearly:
            m.add_seasonality(name="yearly", period=365.25, fourier_order=fourier_yearly)
        m.fit(d)
        fut = m.make_future_dataframe(periods=horizon)
        fc = m.predict(fut)[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": date_col})
        return s, fc, "prophet", m
    ss = s.set_index(date_col)[y_col].asfreq("D").interpolate()
    res = SARIMAX(
        ss,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    pred = res.get_forecast(steps=horizon)
    idx = pd.date_range(ss.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = pd.DataFrame(
        {
            date_col: idx,
            "yhat": pred.predicted_mean.values,
            "yhat_lower": pred.conf_int().iloc[:, 0].values,
            "yhat_upper": pred.conf_int().iloc[:, 1].values,
        }
    )
    return s.reset_index(), fc, "sarimax", None


# ---- Azure helpers ----------------------------------------------------------


def get_azure_client():
    if not HAS_AZURE:
        st.warning("Azure OpenAI client chưa được cài đặt. `pip install openai` (v1.x).")
        return None
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_ver = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    if not endpoint or not api_key:
        st.error("Thiếu AZURE_OPENAI_ENDPOINT hoặc AZURE_OPENAI_API_KEY.")
        return None
    try:
        return AzureOpenAI(api_key=api_key, api_version=api_ver, azure_endpoint=endpoint)
    except Exception as e:
        st.error(f"Lỗi Azure client: {e}")
        return None


def azure_chat(messages, model=None, temperature=0.2, max_tokens=800):
    client = get_azure_client()
    if client is None:
        return None
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Lỗi gọi chat: {e}")
        return None


def azure_chat_stream(messages, model=None, temperature=0.2, max_tokens=800):
    client = get_azure_client()
    if client is None:
        yield from ()
        return
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    try:
        stream = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
            except Exception:
                continue
    except Exception as e:
        st.error(f"Lỗi streaming call: {e}")
        yield from ()


# ---------- Context builder (row-level citations) ----------


class ContextBuilder:
    def __init__(self):
        self.parts = []
        self.records = []
        self.counters = {}

    def _next_rid(self, prefix: str) -> str:
        n = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = n
        return f"{prefix}{n:03d}"

    def add_block(self, df: pd.DataFrame, cols, title: str, prefix: str, max_rows: int = 30):
        if df is None or df.empty:
            return
        d = df.copy()
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        d = d.head(max_rows).reset_index(drop=True)
        rids = [self._next_rid(prefix) for _ in range(len(d))]
        d_out = d.copy()
        d_out.insert(0, "rid", rids)
        for rid, row in zip(rids, d.to_dict(orient="records")):
            rec = {"rid": rid, "table": title}
            rec.update(row)
            self.records.append(rec)
        self.parts.append(f"\n### {title}\n" + d_out.to_csv(index=False))

    def build(self):
        context_text = "\n".join(self.parts)
        citations_df = pd.DataFrame(self.records) if self.records else pd.DataFrame(columns=["rid"])
        return context_text, citations_df


def make_context_for_llm(base_scoped, vnd_offpeak, vnd_normal, vnd_peak, question):
    cb = ContextBuilder()
    if "date" in base_scoped.columns and not base_scoped.empty:
        dmin, dmax = base_scoped["date"].min(), base_scoped["date"].max()
        cb.parts.append(f"Khoảng thời gian: {str(dmin.date())} → {str(dmax.date())}")
    daily = base_scoped.groupby("date", as_index=False)["interval_energy_kwh"].sum()
    cb.add_block(daily, ["date", "interval_energy_kwh"], "Tổng hàng ngày (kWh) [D]", "D", 90)
    if "equipment_name_en" in base_scoped.columns and re.search(r"\bequip|machine|asset\b", question, re.I):
        top = (
            base_scoped.groupby("equipment_name_en", as_index=False)["interval_energy_kwh"]
            .sum()
            .sort_values("interval_energy_kwh", ascending=False)
            .head(20)
        )
        tot = top["interval_energy_kwh"].sum()
        if tot > 0:
            top["share_pct"] = 100 * top["interval_energy_kwh"] / tot
        cb.add_block(
            top,
            ["equipment_name_en", "interval_energy_kwh", "share_pct"],
            "Thiết bị hàng đầu [E]",
            "E",
            20,
        )
    if "meter_location_en" in base_scoped.columns and re.search(r"\bmeter|location|area|line\b", question, re.I):
        ml = (
            base_scoped.groupby("meter_location_en", as_index=False)["interval_energy_kwh"]
            .sum()
            .sort_values("interval_energy_kwh", ascending=False)
            .head(20)
        )
        cb.add_block(
            ml,
            ["meter_location_en", "interval_energy_kwh"],
            "Vị trí đồng hồ hàng đầu [M]",
            "M",
            20,
        )
    if "period_std" in base_scoped.columns and base_scoped["period_std"].notna().any():
        tb = base_scoped.groupby(["date", "period_std"], as_index=False)["interval_energy_kwh"].sum()
        cb.add_block(
            tb,
            ["date", "period_std", "interval_energy_kwh"],
            "Năng lượng theo biểu giá (kWh) [T]",
            "T",
            120,
        )
        if re.search(r"\bcost|vnd|tariff|price|bill|billing\b", question, re.I):
            m = {"Off-peak": vnd_offpeak, "Normal": vnd_normal, "Peak": vnd_peak}
            tbc = tb.copy()
            tbc["period_std"] = tbc["period_std"].map(lambda x: x if x in ["Off-peak", "Normal", "Peak"] else "Unknown")
            tbc["tariff_vnd_kwh"] = tbc["period_std"].map(m).fillna(0.0)
            tbc["cost_vnd"] = tbc["interval_energy_kwh"] * tbc["tariff_vnd_kwh"]
            dc = tbc.groupby("date", as_index=False)["cost_vnd"].sum()
            cb.add_block(dc, ["date", "cost_vnd"], "Chi phí hàng ngày (VND) [C]", "C", 90)
            tot = tbc.groupby("period_std", as_index=False)["cost_vnd"].sum().sort_values("cost_vnd", ascending=False)
            cb.add_block(tot, ["period_std", "cost_vnd"], "Tổng chi phí theo biểu giá [CT]", "CT", 15)
    raw_cols = [
        c
        for c in [
            "date",
            "equipment_name_en",
            "meter_location_en",
            "period_std",
            "interval_energy_kwh",
        ]
        if c in base_scoped.columns
    ]
    cb.add_block(
        base_scoped.sort_values("date", ascending=False),
        raw_cols,
        "Dữ liệu gần đây [R]",
        "R",
        40,
    )
    return cb.build()


# ---- Chart helpers ----------------------------------------------------------


def get_chart_tables(scope, vnd_offpeak, vnd_normal, vnd_peak):
    tables = {}
    if "date" in scope.columns:
        tables["daily"] = scope.groupby("date", as_index=False)["interval_energy_kwh"].sum()
    if "equipment_name_en" in scope.columns:
        tables["equipment"] = (
            scope.groupby("equipment_name_en", as_index=False)["interval_energy_kwh"].sum().sort_values("interval_energy_kwh", ascending=False)
        )
    if "meter_location_en" in scope.columns:
        tables["meter"] = (
            scope.groupby("meter_location_en", as_index=False)["interval_energy_kwh"].sum().sort_values("interval_energy_kwh", ascending=False)
        )
    if "period_std" in scope.columns and scope["period_std"].notna().any():
        tables["tariff_daily"] = scope.groupby(["date", "period_std"], as_index=False)["interval_energy_kwh"].sum()
        t = tables["tariff_daily"].copy()
        m = {"Off-peak": vnd_offpeak, "Normal": vnd_normal, "Peak": vnd_peak}
        t["tariff_vnd_kwh"] = t["period_std"].map(m).fillna(0.0)
        t["cost_vnd"] = t["interval_energy_kwh"] * t["tariff_vnd_kwh"]
        tables["cost_daily"] = t.groupby("date", as_index=False)["cost_vnd"].sum()
    schema = {k: list(v.columns) for k, v in tables.items()}
    return tables, schema


def render_chart_from_spec(spec, tables):
    table = spec.get("table")
    typ = spec.get("chart_type", "line")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    agg = spec.get("agg")
    title = spec.get("title")
    if table not in tables:
        st.warning(f"Bảng không xác định '{table}'.")
        return
    df = tables[table].copy()
    if not df.size:
        st.info("Không có dữ liệu để vẽ biểu đồ.")
        return
    if agg and x and y:
        try:
            df = df.groupby(x, as_index=False)[y].agg(agg)
        except Exception:
            pass
    if typ == "line":
        fig = px.line(df, x=x, y=y, color=color, title=title)
    elif typ == "area":
        fig = px.area(df, x=x, y=y, color=color, title=title)
    elif typ == "bar":
        fig = px.bar(df, x=x, y=y, color=color, title=title)
    elif typ == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
    elif typ == "pie":
        if x and y:
            fig = px.pie(df, names=x, values=y, title=title)
        else:
            st.warning("Biểu đồ tròn cần x(names) và y(values).")
            return
    elif typ == "treemap":
        path_cols = [c for c in [x, color] if c]
        value = y or (df.columns[1] if len(df.columns) > 1 else None)
        if not value:
            st.warning("Treemap cần một cột giá trị.")
            return
        fig = px.treemap(df, path=path_cols or [x], values=value, title=title)
    else:
        st.warning(f"Loại biểu đồ không hỗ trợ '{typ}'.")
        return

    # Force white text on all chart elements
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=460,
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        hoverlabel=dict(font=dict(color="white")),
    )
    fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
    fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))

    st.plotly_chart(fig, use_container_width=True)


# ----- AI insight helpers ----------------------------------------------------


def ai_box(title: str, text: str):
    """Render AI output in a black 'terminal' style box."""
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div class='ai-box'><div class='ai-title'>Thông tin chi tiết — {title}</div>{safe}</div>",
        unsafe_allow_html=True,
    )


def ai_insights(title: str, context_dict: dict):
    """Call Azure OpenAI and render a concise insights block."""
    if not HAS_AZURE:
        st.info("Azure OpenAI client không tìm thấy. Cài đặt: `pip install openai` (v1.x).")
        return
    safe_payload = json.dumps(json_ready(context_dict), ensure_ascii=False)
    prompt = (
        "Tóm tắt những điểm chính thành 3-6 điểm sử dụng dữ liệu JSON này. "
        "Hãy cụ thể và bao gồm kWh và VND khi có liên quan. Giữ dưới 120 từ.\n"
        f"{safe_payload[:15000]}"
    )
    msg = [
        {"role": "system", "content": "Bạn là một trợ lý phân tích ngắn gọn."},
        {"role": "user", "content": prompt},
    ]
    txt = azure_chat(msg, temperature=0.1, max_tokens=220)
    if txt:
        ai_box(title, txt)


# ---------------- LOAD DATA ---------------------------------------------------
dfs, missing = load_all(DATA_DIR)
if "long" not in dfs:
    st.error("Thiếu CSV: energy_gold_detail_long.csv")
    st.stop()
for k in dfs:
    dfs[k] = coerce_date(dfs[k], "date")
base = dfs["long"].copy()
if "interval_energy_kwh" not in base.columns:
    base["interval_energy_kwh"] = 0.0
if "period_std" not in base.columns:
    base["period_std"] = None
base["interval_energy_kwh"] = base["interval_energy_kwh"].fillna(0.0)

# ---------------- SIDEBAR (logos + filters) ----------------------------------
logo1 = to_data_uri(LOGO1_PATH)
logo2 = to_data_uri(LOGO2_PATH)
with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-card">
          <div class="brand-logos">
            {f"<img src='{logo1}'>" if logo1 else ""}
            {f"<img src='{logo2}'>" if logo2 else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

base, dr = daterange(base, "date")
if "equipment_name_en" in base.columns:
    base = ms_filter(base, "equipment_name_en", "Thiết bị")
if "meter_location_en" in base.columns:
    base = ms_filter(base, "meter_location_en", "Vị trí đồng hồ")

st.sidebar.write("---")
st.sidebar.caption("Biểu giá (VND/kWh)")
vnd_offpeak = st.sidebar.number_input("Giờ thấp điểm", min_value=0.0, value=0.0, step=100.0, format="%.1f")
vnd_normal = st.sidebar.number_input("Giờ bình thường", min_value=0.0, value=0.0, step=100.0, format="%.1f")
vnd_peak = st.sidebar.number_input("Giờ cao điểm", min_value=0.0, value=0.0, step=100.0, format="%.1f")

# ---------------- HERO --------------------------------------------------------
st.markdown(f"## {APP_TITLE}")
st.markdown(
    "<div class='subtle'>Giám sát, dự báo và giải thích sử dụng điện trong toàn bộ nhà máy của bạn.</div>",
    unsafe_allow_html=True,
)

# ---------------- TABS --------------------------------------------------------
(
    tab_overview,
    tab_equipment,
    tab_tariff,
    tab_forecast,
    tab_anom,
    tab_quality,
    tab_nlp,
) = st.tabs(
    [
        "Tổng quan",
        "Thiết bị",
        "Biểu giá & Chi phí",
        "Dự báo",
        "Bất thường",
        "Chất lượng dữ liệu",
        "Hỏi dữ liệu",
    ]
)

# ---------- OVERVIEW ----------
with tab_overview:
    daily = base.groupby("date", as_index=False)["interval_energy_kwh"].sum()
    total = float(daily["interval_energy_kwh"].sum())
    avg_daily = float(daily["interval_energy_kwh"].mean() or 0.0)
    c1, c2 = st.columns(2)
    with c1:
        kpi(total, "Tổng năng lượng (kWh)")
    with c2:
        kpi(avg_daily, "Trung bình mỗi ngày (kWh)")

    st.markdown("#### Năng lượng hàng ngày")
    daily_plot, clipped, hi = cap_for_plot(daily, "interval_energy_kwh")
    fig = px.line(daily_plot, x="date", y="interval_energy_kwh", markers=True)
    if hi is not None:
        fig.update_yaxes(range=[0, hi])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=460, font=dict(color="white"), hoverlabel=dict(font=dict(color="white")))
    fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
    fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)
    if clipped > 0:
        st.caption("Trục y của biểu đồ được giới hạn để dễ đọc; tổng số sử dụng dữ liệu đầy đủ.")

    if st.button("Tạo thông tin chi tiết với AI tạo sinh", key="ai_overview_btn"):
        ai_insights(
            "Tổng quan",
            {
                "series_daily": daily.tail(180),
                "totals": {"total_kwh": total, "avg_day_kwh": avg_daily},
            },
        )

# ---------- EQUIPMENT ----------
with tab_equipment:
    if "equipment_name_en" not in base.columns:
        st.info("Cột thiết bị không có sẵn trong dữ liệu.")
    else:
        st.markdown("#### Thiết bị hàng đầu theo năng lượng")
        top = (
            base.groupby("equipment_name_en", as_index=False)["interval_energy_kwh"]
            .sum()
            .sort_values("interval_energy_kwh", ascending=False)
            .head(30)
        )
        fig = px.bar(top, x="equipment_name_en", y="interval_energy_kwh")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=460, font=dict(color="white"), hoverlabel=dict(font=dict(color="white")))
        fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Xu hướng hàng ngày — lựa chọn")
        eq = st.selectbox(
            "Thiết bị",
            ["(Tất cả)"] + sorted(base["equipment_name_en"].dropna().unique().tolist()),
        )
        scope = base if eq == "(Tất cả)" else base[base["equipment_name_en"] == eq]
        d = scope.groupby("date", as_index=False)["interval_energy_kwh"].sum()
        d_plot, _, hi = cap_for_plot(d, "interval_energy_kwh")
        fig = px.area(d_plot, x="date", y="interval_energy_kwh")
        if hi is not None:
            fig.update_yaxes(range=[0, hi])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=460, font=dict(color="white"), hoverlabel=dict(font=dict(color="white")))
        fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        if "meter_location_en" in scope.columns:
            st.markdown("#### Đóng góp vị trí đồng hồ")
            bm = (
                scope.groupby("meter_location_en", as_index=False)["interval_energy_kwh"]
                .sum()
                .sort_values("interval_energy_kwh", ascending=False)
                .head(30)
            )
            fig = px.bar(bm, x="meter_location_en", y="interval_energy_kwh")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=460, font=dict(color="white"), hoverlabel=dict(font=dict(color="white")))
            fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
            fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Tạo thông tin chi tiết với AI tạo sinh", key="ai_eq_btn"):
            ai_insights(
                "Thiết bị",
                {"top_equipment": top, "selected": eq, "selected_series": d.tail(120)},
            )

# ---------- TARIFFS & COST ----------
with tab_tariff:
    st.markdown("#### Năng lượng được gắn thẻ biểu giá")
    if "period_std" in base.columns and base["period_std"].notna().any():
        tb = base.groupby(["date", "period_std"], as_index=False)["interval_energy_kwh"].sum()
        tb["period_std"] = tb["period_std"].fillna("Không xác định")
        fig = px.area(
            tb,
            x="date",
            y="interval_energy_kwh",
            color="period_std",
            category_orders={"period_std": ["Off-peak", "Normal", "Peak", "Không xác định"]},
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=460,
            legend_title_text="",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            hoverlabel=dict(font=dict(color="white")),
        )
        fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nhãn biểu giá không có trong dữ liệu.")

    st.markdown("#### Chi phí ước tính (VND)")
    if vnd_offpeak + vnd_normal + vnd_peak == 0:
        st.caption("Nhập biểu giá trong thanh bên trái để tính chi phí.")
        daily_cost = pd.DataFrame()
    else:
        tb = base[["date", "period_std", "interval_energy_kwh"]].copy()
        tb["period_std"] = tb["period_std"].map(lambda x: x if x in ["Off-peak", "Normal", "Peak"] else "Không xác định")
        m = {"Off-peak": vnd_offpeak, "Normal": vnd_normal, "Peak": vnd_peak}
        tb["tariff_vnd_kwh"] = tb["period_std"].map(m).fillna(0.0)
        tb["cost_vnd"] = tb["interval_energy_kwh"] * tb["tariff_vnd_kwh"]
        daily_cost = tb.groupby(["date", "period_std"], as_index=False)["cost_vnd"].sum()
        fig = px.area(daily_cost, x="date", y="cost_vnd", color="period_std")
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=460,
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            hoverlabel=dict(font=dict(color="white")),
        )
        fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
        totals = daily_cost.groupby("period_std", as_index=False)["cost_vnd"].sum().sort_values("cost_vnd", ascending=False)
        st.dataframe(totals, use_container_width=True)

        if st.button("Tạo thông tin chi tiết với AI tạo sinh", key="ai_tariff_btn"):
            ai_insights(
                "Biểu giá & Chi phí",
                {
                    "daily_tariff": tb.tail(120),
                    "daily_cost": daily_cost.tail(120),
                    "totals": totals,
                },
            )

# ---------- FORECASTING ----------
with tab_forecast:
    st.markdown("#### Cấu hình mô hình")
    tgt = st.selectbox("Chuỗi mục tiêu", ["Tổng địa điểm", "Theo thiết bị", "Theo vị trí đồng hồ"])
    scope = base.copy()
    if tgt == "Theo thiết bị" and "equipment_name_en" in scope.columns:
        eq = st.selectbox("Thiết bị", sorted(scope["equipment_name_en"].dropna().unique().tolist()))
        scope = scope[scope["equipment_name_en"] == eq]
    elif tgt == "Theo vị trí đồng hồ" and "meter_location_en" in scope.columns:
        ml = st.selectbox(
            "Vị trí đồng hồ",
            sorted(scope["meter_location_en"].dropna().unique().tolist()),
        )
        scope = scope[scope["meter_location_en"] == ml]

    weekly = st.checkbox("Tính mùa vụ hàng tuần", value=True)
    yearly = st.checkbox("Tính mùa vụ hàng năm", value=True)
    fw = st.slider("Fourier (hàng tuần)", 3, 15, 6)
    fy = st.slider("Fourier (hàng năm)", 5, 25, 10)
    cp = st.slider("Thang điểm trước điểm thay đổi", 0.01, 0.5, 0.05)
    sp = st.slider("Thang điểm trước tính mùa vụ", 1.0, 20.0, 10.0)
    horizon = st.slider("Chân trời dự báo (ngày)", 7, 180, 60)
    series = scope.groupby("date", as_index=False)["interval_energy_kwh"].sum().rename(columns={"interval_energy_kwh": "energy_kwh"})

    colA, colB, colC = st.columns([1, 1, 1])
    run_fc = colA.button("Chạy dự báo")
    show_comp = colB.button("Tính mùa vụ (Prophet)")
    show_out = colC.button("Ngoại lệ (Prophet)")

    if run_fc:
        try:
            s, fc, model, m = train_forecast(
                series,
                y_col="energy_kwh",
                horizon=horizon,
                use_prophet=USE_PROPHET,
                weekly=weekly,
                yearly=yearly,
                fourier_weekly=fw,
                fourier_yearly=fy,
                changepoint_prior=cp,
                seasonality_prior=sp,
            )
            st.markdown(
                f"<div class='status-ok'>Mô hình: {model.upper()}</div>",
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["date"], y=s["energy_kwh"], mode="lines+markers", name="Lịch sử"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines", name="Dự báo"))
            fig.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc["yhat_upper"],
                    mode="lines",
                    name="Giới hạn trên",
                    line=dict(dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc["yhat_lower"],
                    mode="lines",
                    name="Giới hạn dưới",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=460,
                font=dict(color="white"),
                legend=dict(font=dict(color="white")),
                hoverlabel=dict(font=dict(color="white")),
            )
            fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
            fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi dự báo: {e}")

    if show_comp:
        if not USE_PROPHET:
            st.warning("Prophet không có sẵn.")
        else:
            try:
                _, _, _, m = train_forecast(
                    series,
                    y_col="energy_kwh",
                    horizon=0,
                    use_prophet=True,
                    weekly=weekly,
                    yearly=yearly,
                    fourier_weekly=fw,
                    fourier_yearly=fy,
                    changepoint_prior=cp,
                    seasonality_prior=sp,
                )
                comp_fig = m.plot_components(m.predict(m.make_future_dataframe(periods=0)))
                st.pyplot(comp_fig, clear_figure=True)
            except Exception as e:
                st.error(f"Lỗi tính mùa vụ: {e}")

    if show_out:
        if not USE_PROPHET:
            st.warning("Prophet không có sẵn.")
        else:
            try:
                s = series.rename(columns={"date": "ds", "energy_kwh": "y"}).dropna()
                m = Prophet(changepoint_prior_scale=cp, seasonality_prior_scale=sp)
                if weekly:
                    m.add_seasonality(name="weekly", period=7, fourier_order=fw)
                if yearly:
                    m.add_seasonality(name="yearly", period=365.25, fourier_order=fy)
                m.fit(s)
                pred = m.predict(s[["ds"]])
                dfc = s.copy()
                dfc["yhat"] = pred["yhat"].values
                dfc["resid"] = dfc["y"] - dfc["yhat"]
                z = (dfc["resid"] - dfc["resid"].mean()) / (dfc["resid"].std(ddof=0) + 1e-9)
                dfc["z"] = z
                dfc["is_outlier"] = z.abs() >= 3.0
                st.dataframe(
                    dfc.sort_values("z", ascending=False).head(30),
                    use_container_width=True,
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfc["ds"], y=dfc["y"], name="Thực tế", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=dfc["ds"], y=dfc["yhat"], name="Prophet fit", mode="lines"))
                out = dfc[dfc["is_outlier"]]
                fig.add_trace(
                    go.Scatter(
                        x=out["ds"],
                        y=out["y"],
                        mode="markers",
                        name="Ngoại lệ",
                        marker=dict(size=9, symbol="x"),
                    )
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=460,
                    font=dict(color="white"),
                    legend=dict(font=dict(color="white")),
                    hoverlabel=dict(font=dict(color="white")),
                )
                fig.update_xaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
                fig.update_yaxes(tickfont=dict(color="white"), titlefont=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi phân tích ngoại lệ: {e}")

# ---------- ANOMALIES ----------
with tab_anom:
    if "equipment_name_en" not in base.columns:
        st.info("Cột thiết bị không có sẵn trong dữ liệu.")
    else:
        daily_eq = (
            base.groupby(["date", "equipment_name_en"], as_index=False)["interval_energy_kwh"]
            .sum()
            .rename(columns={"interval_energy_kwh": "energy_kwh"})
        )
        thr = st.slider("Ngưỡng Z", 2.0, 5.0, 3.0, 0.1)
        z = daily_eq.copy()
        z["z"] = z.groupby("equipment_name_en")["energy_kwh"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
        z["is_outlier"] = z["z"].abs() >= thr
        st.dataframe(z.sort_values("z", ascending=False).head(80), use_container_width=True)

        if st.button("Tạo thông tin chi tiết với AI tạo sinh", key="ai_anom_btn"):
            ai_insights(
                "Bất thường",
                {"top_outliers": z.sort_values("z", ascending=False).head(30)},
            )

# ---------- DATA QUALITY ----------
with tab_quality:
    miss = base.isna().mean().sort_values(ascending=False)
    st.dataframe(miss.to_frame("tỷ_lệ_thiếu"), use_container_width=True)
    st.markdown("#### Mẫu (50 dòng đầu tiên)")
    st.write(base.head(50))

# ---------- ASK THE DATA (NLP) ----------
with tab_nlp:
    st.markdown("#### Hỏi đáp bằng ngôn ngữ tự nhiên")
    if not HAS_AZURE:
        st.warning("Azure OpenAI client không tìm thấy. Cài đặt: `pip install openai` (v1.x).")

    with st.expander("Thông tin xác thực Azure (ghi đè phiên)"):
        ep = st.text_input("Endpoint", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        key = st.text_input("API Key", type="password")
        dep = st.text_input("Deployment", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))
        ver = st.text_input("Phiên bản API", os.getenv("OPENAI_API_VERSION", "2024-02-01"))
        if st.button("Sử dụng các thông tin này"):
            os.environ["AZURE_OPENAI_ENDPOINT"] = ep.strip()
            os.environ["AZURE_OPENAI_API_KEY"] = key.strip()
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = dep.strip()
            os.environ["OPENAI_API_VERSION"] = ver.strip()
            st.success("Thông tin xác thực phiên đã được đặt. Hỏi lại bên dưới.")

    scope = base.copy()
    q = st.text_area(
        "Câu hỏi",
        placeholder="ví dụ: Thiết bị nào tiêu thụ nhiều nhất tháng trước?",
        height=90,
    )
    show_ctx = st.checkbox("Hiển thị ngữ cảnh được gửi đến mô hình (debug)")
    ask_btn = st.button("Hỏi (streaming)")

    chart_tables, chart_schema = get_chart_tables(scope, vnd_offpeak, vnd_normal, vnd_peak)

    if ask_btn and q.strip():
        try:
            context_text, citations_df = make_context_for_llm(scope, vnd_offpeak, vnd_normal, vnd_peak, q)
        except Exception as e:
            st.error(f"Lỗi xây dựng ngữ cảnh: {e}")
            st.stop()

        system_prompt = (
            "Bạn là một đồng hành phân tích. Chỉ sử dụng ngữ cảnh CSV được cung cấp. "
            "Bao gồm số rõ ràng với đơn vị (kWh/VND). "
            "Khi trích dẫn một dòng, thêm trích dẫn như [RID:<rid>]. Nếu dữ liệu thiếu, nêu chính xác những gì."
        )
        user_prompt = (
            f"CÂU HỎI:\n{q}\n\nNGỮ CẢNH:\n{context_text}\n\n"
            "Trả về 3-7 điểm ngắn gọn với số liệu chính (kWh/VND), bao gồm trích dẫn [RID:<rid>], "
            "và một khuyến nghị ngắn gọn nếu hữu ích."
        )

        if show_ctx:
            with st.expander("Xem trước ngữ cảnh"):
                st.code(user_prompt[:8000], language="markdown")

        placeholder = st.empty()
        answer = ""

        def render_stream(txt):
            safe = html.escape(txt).replace("\n", "<br>")
            placeholder.markdown(f"<div class='ai-box'>{safe}</div>", unsafe_allow_html=True)

        with st.spinner("Đang suy nghĩ…"):
            for token in azure_chat_stream(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                temperature=0.2,
                max_tokens=1000,
            ):
                answer += token
                render_stream(answer)

        st.markdown("##### Trích dẫn")
        rids = re.findall(r"\[RID:([A-Za-z]+[0-9]+)\]", answer)
        if citations_df is not None and not citations_df.empty and rids:
            st.dataframe(
                citations_df[citations_df["rid"].isin(sorted(set(rids)))],
                use_container_width=True,
            )
        else:
            st.info("Không phát hiện trích dẫn nào trong câu trả lời.")

        st.markdown("##### Tạo biểu đồ")
        if st.button("Tạo biểu đồ từ câu hỏi này"):
            schema_text = json.dumps(chart_schema, indent=2)
            chart_prompt = (
                "Đề xuất một biểu đồ để trả lời câu hỏi của người dùng bằng cách sử dụng các bảng có sẵn.\n"
                'Trả lời chỉ bằng JSON nghiêm ngặt: {"table":..., "chart_type":"line|area|bar|scatter|pie|treemap", '
                '"x":"...", "y":"...", "color":null|"...", "agg":null|"sum"|"mean", "title":"..."}\n'
                f"Bảng và cột:\n{schema_text}"
            )
            spec_text = azure_chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một người lập kế hoạch trực quan hóa dữ liệu. Chỉ xuất ra JSON nghiêm ngặt.",
                    },
                    {"role": "user", "content": f"Câu hỏi: {q}\n{chart_prompt}"},
                ],
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                temperature=0.0,
                max_tokens=300,
            )
            try:
                spec = json.loads(spec_text)
                st.code(json.dumps(spec, indent=2), language="json")
                render_chart_from_spec(spec, chart_tables)
            except Exception as e:
                st.error(f"Không thể phân tích spec biểu đồ. Raw: {spec_text}\nLỗi: {e}")
