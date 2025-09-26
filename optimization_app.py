# Hospital Optimization Studio — Exec-Ready, Easy Deploy (No Gurobi)
# Modules:
#  1) Price Prediction: learn expected billing per case (+ explainers)
#  2) Case Mix Optimizer (MILP via PuLP/CBC): minimize predicted cost under policy constraints
#  3) Staffing Optimizer: RN targets from admissions forecast (HW/SARIMAX)
#  4) Trends & Explainability
#
# Design choices:
#  - No sidebar; full-width, horizontal tabs; Sutter/Kaiser-friendly palette
#  - Optional OpenAI for an AI Analyst (falls back gracefully if key missing)
#  - Fast, portable stack (no Gurobi, no Prophet/XGBoost/shap dependencies)
#
# Data:
#  - Default: GitHub CSV
#  - Auto-load /mnt/data/modified_healthcare_dataset.csv if present
#  - Upload override supported in-page

import os, json, textwrap, re, math
from datetime import datetime
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize, LpInteger, LpBinary,
    LpStatus, PULP_CBC_CMD
)

# ---------------- CONFIG / THEME ----------------
st.set_page_config(page_title="Hospital Optimization Studio", layout="wide", page_icon="🏥")

RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"

# Calm clinical theme (Sutter/Kaiser friendly blues/teals) + narrative cards
st.markdown("""
<style>
:root{
  --ink:#0b2740; --pri:#0F4C81; --teal:#159E99; --sub:#5b6b7a; --bg:#f7f9fb;
  --ok:#14A38B; --warn:#F59E0B; --alert:#EF4444;
}
html, body, [class^="css"]  {background-color: var(--bg);}
.block-container{max-width:1500px;padding-top:12px}
h1,h2,h3{font-weight:700;color:var(--ink)}
.stTabs [data-baseweb="tab-list"] { gap: 6px }
.stTabs [data-baseweb="tab"]{
  background: white; padding: 8px 14px; border-radius: 10px; border: 1px solid #e8eef5;
}
.stTabs [aria-selected="true"]{
  background: #e8f2ff; border-color:#c8defc;
}
a {color:var(--teal)}
.stButton>button{background:var(--pri);color:#fff;border-radius:10px;border:0}
.badge{display:inline-block;padding:.25rem .55rem;border-radius:.5rem;background:#eef3f7;color:var(--ink);margin-right:.3rem}
.small{color:var(--sub);font-size:0.92rem}
hr{border-top:1px solid #e9eef3}
.kpi{background:white;border:1px solid #e8eef5;border-radius:16px;padding:14px}
.card{
  background:white;border:1px solid #e8eef5;border-radius:14px;padding:14px;margin-bottom:10px
}
.card h4{margin:0 0 6px 0}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;
      border:1px solid #dde6ef;background:#f5f8fb;color:#334155;margin-right:6px}
.table-note{font-size:12px;color:#64748b;margin-top:6px}
.copybox{
  white-space:pre-wrap;background:#0b2740;color:#e6eef7;border-radius:12px;padding:12px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Optimization Studio")
st.caption("Price Prediction • Case Mix Optimization • Staffing Targets • Trends • AI Analyst")

# ---------------- HELPERS ----------------
def _markdown_escape(text: str) -> str:
    return re.sub(r'([*_`])', r'\\\1', str(text))

def ensure_columns(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Ensure DataFrame has all columns in 'cols', creating missing ones as NaN."""
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        for m in missing:
            frame[m] = np.nan
    return frame[cols]

def safe_predict(pipeline: Pipeline, X_like: pd.DataFrame, cols_needed: Sequence[str]) -> np.ndarray:
    """
    Predict robustly on uploaded data with missing columns.
    Uses imputers inside the pipeline; no NaNs reach StandardScaler.
    """
    X_aligned = ensure_columns(X_like.copy(), cols_needed)
    return pipeline.predict(X_aligned)

def infer_ts_granularity(idx: pd.DatetimeIndex) -> Tuple[str, Optional[int]]:
    """
    Infer sensible resample frequency and seasonality.
    Returns (freq, seasonal_periods or None).
    - Daily if median gap <= 1.5 days (seasonality 7)
    - Weekly if median gap <= 10 days (seasonality 52 if enough points, else None)
    - Monthly otherwise (seasonality 12 if enough points, else None)
    """
    if len(idx) < 3:
        return "D", None
    diffs = np.diff(np.sort(idx.values).astype("datetime64[D]").astype("int64"))
    med_gap = np.median(diffs) if len(diffs) else 1
    if med_gap <= 1.5:
        return "D", 7
    elif med_gap <= 10:
        return "W", 52
    else:
        return "M", 12

def build_timeseries(data: pd.DataFrame, metric: str, freq: Optional[str] = None) -> pd.DataFrame:
    if "admit_date" not in data.columns:
        return pd.DataFrame(columns=["ds","y"])
    idx_df = data.set_index("admit_date").sort_index()
    if freq is None:
        # choose based on data density
        inferred_freq, _ = infer_ts_granularity(idx_df.index)
    else:
        inferred_freq = freq

    if metric == "intake":
        s = idx_df.assign(_one=1)["_one"].resample(inferred_freq).sum().fillna(0.0)
    elif metric == "billing_amount" and "billing_amount" in idx_df:
        s = idx_df["billing_amount"].resample(inferred_freq).sum().ffill().fillna(0.0)
    elif metric == "length_of_stay" and "length_of_stay" in idx_df:
        s = idx_df["length_of_stay"].resample(inferred_freq).mean().ffill().fillna(0.0)
    else:
        return pd.DataFrame(columns=["ds","y"])
    return pd.DataFrame({"ds": s.index, "y": s.values})

def hw_forecast(s: pd.Series, horizon: int, seasonal_periods: Optional[int]) -> np.ndarray:
    """
    Holt-Winters with guardrails:
    - Use additive seasonality only if enough history: >= 3*seasonal_periods.
    - Otherwise trend-only.
    """
    try:
        if seasonal_periods and len(s) >= 3*seasonal_periods:
            m = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
        else:
            m = ExponentialSmoothing(s, trend="add").fit()
    except Exception:
        m = ExponentialSmoothing(s, trend="add").fit()
    return m.forecast(horizon)

def sarimax_forecast(s: pd.Series, horizon: int, seasonal_periods: Optional[int]) -> np.ndarray:
    seas = seasonal_periods if seasonal_periods else 0
    seas = max(seas, 0)
    if seas >= 2:
        order = (1,1,1); seasonal_order = (1,1,1,seas)
    else:
        order = (1,1,1); seasonal_order = (0,0,0,0)
    model = SARIMAX(
        s, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    return model.forecast(steps=horizon)

def admissions_forecast(data: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    ts = build_timeseries(data, "intake")  # auto-infers frequency
    if ts.empty:
        return pd.DataFrame(columns=["ds","yhat"])
    s = ts.set_index("ds")["y"]
    # ensure fixed freq and fill forward
    inferred_freq, seasonal_periods = infer_ts_granularity(s.index)
    s = s.asfreq(inferred_freq).ffill()

    # If very short or flat, repeat last value
    if len(s) < 5 or s.nunique() <= 1:
        last = float(s.iloc[-1]) if len(s) else 0.0
        ds = pd.date_range(s.index.max() + pd.tseries.frequencies.to_offset(inferred_freq), periods=horizon, freq=inferred_freq)
        return pd.DataFrame({"ds": ds, "yhat": np.repeat(last, horizon)})

    try:
        fc = hw_forecast(s, horizon, seasonal_periods)
    except Exception:
        try:
            fc = sarimax_forecast(s, horizon, seasonal_periods)
        except Exception:
            last = float(s.iloc[-1]); fc = np.repeat(last, horizon)

    ds = pd.date_range(s.index.max() + pd.tseries.frequencies.to_offset(inferred_freq), periods=horizon, freq=inferred_freq)
    return pd.DataFrame({"ds": ds, "yhat": np.asarray(fc)})

# ---------------- DATA LOAD & FE ----------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # Priority: local mount -> RAW
    try:
        local = "/mnt/data/modified_healthcare_dataset.csv"
        if os.path.exists(local):
            df = pd.read_csv(local)
        else:
            df = pd.read_csv(RAW_URL)
    except Exception:
        df = pd.read_csv(RAW_URL)

    # Normalize columns
    rename_map = {
        "Billing Amount": "billing_amount",
        "Date of Admission": "admit_date",
        "Discharge Date": "discharge_date",
        "Length of Stay": "length_of_stay",
        "Medical Condition": "condition",
        "Admission Type": "admission_type",
        "Insurance Provider": "insurer",
        "Hospital": "hospital",
        "Doctor": "doctor",
        "Medication": "medication",
        "Gender": "gender",
        "Age": "age",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Types & basic hygiene
    if "admit_date" in df.columns:
        df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
    if "discharge_date" in df.columns:
        df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    for c in ["billing_amount", "length_of_stay", "age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep rows with admit_date & billing
    df = df.dropna(subset=["admit_date", "billing_amount"]).copy()

    # Feature Engineering
    df["dow"] = df["admit_date"].dt.weekday
    df["month"] = df["admit_date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Compute LOS if missing/all-NaN and dates exist, and guard negatives
    if "discharge_date" in df.columns and "admit_date" in df.columns:
        if "length_of_stay" not in df.columns or df["length_of_stay"].isna().all():
            df["length_of_stay"] = (df["discharge_date"] - df["admit_date"]).dt.days
        df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
        # set negative LOS to NaN (invalid discharge/admit ordering)
        df.loc[df["length_of_stay"] < 0, "length_of_stay"] = np.nan

    # Robust long-stay flag (handle all-NaN)
    if "length_of_stay" in df.columns and df["length_of_stay"].notna().sum() > 0:
        q95 = df["length_of_stay"].quantile(0.95)
        df["is_longstay"] = (df["length_of_stay"] > q95).astype(int)
    else:
        df["is_longstay"] = 0

    # Categorical hygiene
    for c in ["gender","insurer","hospital","doctor","condition","medication","admission_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    # Light anomaly flag (univariate, but safer threshold)
    try:
        if "billing_amount" in df.columns and len(df) >= 200:
            iso = IsolationForest(contamination=0.03, random_state=42)
            df["anomaly_flag"] = (iso.fit_predict(df[["billing_amount"]]) == -1).astype(int)
        else:
            df["anomaly_flag"] = 0
    except Exception:
        df["anomaly_flag"] = 0

    return df

df = load_data()

with st.expander("📁 Data source (optional override)"):
    up = st.file_uploader("Upload CSV to override default", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            # Quick normalization if schema resembles the expected one
            if "Date of Admission" in df.columns and "Billing Amount" in df.columns:
                df = df.rename(columns={
                    "Date of Admission":"admit_date", "Discharge Date":"discharge_date",
                    "Billing Amount":"billing_amount", "Length of Stay":"length_of_stay",
                    "Medical Condition":"condition", "Admission Type":"admission_type",
                    "Insurance Provider":"insurer", "Hospital":"hospital", "Doctor":"doctor",
                    "Medication":"medication", "Gender":"gender", "Age":"age",
                })
                df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
                if "discharge_date" in df.columns:
                    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
            st.success(f"Loaded {len(df):,} rows from upload.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# ---------------- OpenAI (optional) ----------------
def get_openai_client():
    key = (
        st.secrets.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or st.session_state.get("OPENAI_API_KEY")
    )
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None

# ---------- EXEC/ANALYST STORYTELLING UI ----------
def _df_from_rows(rows, columns):
    if not rows:
        return pd.DataFrame(columns=columns)
    if isinstance(rows[0], dict):
        try:
            return pd.DataFrame(rows)[columns]
        except Exception:
            return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=columns)

def _render_exec_cards(headline, what, so_what, now_what):
    st.markdown(f"""
    <div class="card"><h4>Headline</h4><div>{_markdown_escape(headline)}</div></div>
    <div class="card"><h4>What happened</h4><div>{_markdown_escape(what)}</div></div>
    <div class="card"><h4>Why it matters</h4><div>{_markdown_escape(so_what)}</div></div>
    <div class="card"><h4>What we should do</h4><div>{_markdown_escape(now_what)}</div></div>
    """, unsafe_allow_html=True)

def _render_performance_table(perf_rows):
    dfp = _df_from_rows(perf_rows, ["metric","value","use_case","owner"])
    if not dfp.empty:
        st.markdown("**Performance & Use Cases**")
        st.dataframe(dfp, use_container_width=True, hide_index=True)
        st.markdown('<div class="table-note">Numbers reflect current selection/model run.</div>', unsafe_allow_html=True)
    else:
        st.info("No performance/use case rows returned.")

def _render_recommendations(recs):
    dfr = _df_from_rows(recs, ["action","owner","sla","rationale","confidence"])
    if dfr.empty:
        st.info("No recommendations returned.")
        return
    if "confidence" in dfr.columns:
        dfr["confidence"] = dfr["confidence"].apply(lambda x: f"{round(float(x)*100):.0f}%" if pd.notna(x) else "")
    st.markdown("**Recommendations (Owners & SLAs)**")
    st.dataframe(dfr, use_container_width=True, hide_index=True)

def _download_narrative(json_obj, filename_prefix="narrative"):
    md_lines = []
    md_lines.append(f"# {_markdown_escape(json_obj.get('headline','Executive Summary'))}")
    for k, title in [("what","What happened"),("so_what","Why it matters"),("now_what","What we should do")]:
        if json_obj.get(k):
            md_lines.append(f"## {title}\n{_markdown_escape(json_obj[k])}")
    if json_obj.get("performance_rows"):
        md_lines.append("## Performance & Use Cases")
        md_lines.append("| Metric | Value | Use Case | Owner |")
        md_lines.append("|---|---:|---|---|")
        for r in json_obj["performance_rows"]:
            metric = _markdown_escape(str(r.get("metric",""))); value = _markdown_escape(str(r.get("value","")))
            use_case = _markdown_escape(str(r.get("use_case",""))); owner = _markdown_escape(str(r.get("owner","")))
            md_lines.append(f"| {metric} | {value} | {use_case} | {owner} |")
    if json_obj.get("recommendations"):
        md_lines.append("## Recommendations")
        for i, r in enumerate(json_obj["recommendations"], 1):
            act = _markdown_escape(str(r.get("action",""))); own = _markdown_escape(str(r.get("owner","")))
            sla = _markdown_escape(str(r.get("sla",""))); rat = _markdown_escape(str(r.get("rationale","")))
            conf = r.get("confidence", "")
            conf = f" (confidence: {round(float(conf)*100):.0f}%)" if conf not in ("", None) else ""
            md_lines.append(f"- **{i}. {act}** — Owner: {own}; SLA: {sla}{conf}. {rat}")
    md = "\n\n".join(md_lines)
    st.download_button("⬇️ Download narrative (Markdown)", data=md, file_name=f"{filename_prefix}.md", mime="text/markdown")
    st.download_button("⬇️ Download narrative (JSON)", data=json.dumps(json_obj, indent=2), file_name=f"{filename_prefix}.json", mime="application/json")
    with st.expander("Preview (Markdown)"):
        st.markdown(f"<div class='copybox'>{_markdown_escape(md)}</div>", unsafe_allow_html=True)

def ai_write(section_title: str, payload: dict):
    """
    Executive/Analyst storytelling panel.
    - If OpenAI is available: requests STRICT JSON and renders rich UI.
    - Else: deterministic summary with the same layout.
    """
    client = get_openai_client()
    colL, colR = st.columns([1,2])
    use_ai = colL.checkbox(f"Use AI for {section_title}", value=(client is not None), key=f"ai_{section_title}")
    mode = colR.radio("Audience", ["Executive Summary", "Analyst Deep-Dive"], horizontal=True, key=f"mode_{section_title}")

    panel = st.container()
    with panel:
        st.markdown(f"<div class='pill'>Section: {section_title}</div>", unsafe_allow_html=True)

        if use_ai and client:
            schema_hint = {
                "headline": "string (1 sentence)",
                "what": "string (2-4 sentences)",
                "so_what": "string (2-4 sentences)",
                "now_what": "string (2-4 sentences)",
                "performance_rows": [
                    {"metric": "string", "value": "string/number", "use_case": "string", "owner": "string"}
                ],
                "recommendations": [
                    {"action": "string", "owner": "string", "sla": "e.g., '14 days'", "rationale": "string (1-2 lines)", "confidence": "0.0-1.0"}
                ],
                "risks": ["string", "string"]
            }

            prompt = textwrap.dedent(f"""
            You are a healthcare operations product analyst AND UX writer. Return STRICT JSON only (no prose),
            following this schema exactly:
            {json.dumps(schema_hint, indent=2)}
            Guidance:
            - Tone depends on "audience": {"'executive'" if mode=="Executive Summary" else "'analyst'"}.
            - Use the JSON below as the ONLY source of truth. If a value is missing, be conservative.
            - Keep total words ~140–220.
            - Performance rows should be concise and board-ready.
            - Recommendations must be concrete with owners and SLAs.
            JSON INPUT (source-of-truth):
            {json.dumps(payload, default=str)[:6000]}
            """).strip()

            try:
                rsp = client.chat.completions.create(
                    model=(st.secrets.get("PREFERRED_OPENAI_MODEL") or "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "Respond with STRICT JSON that conforms to the provided schema. No extra text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                raw = rsp.choices[0].message.content
                raw = raw.strip()
                raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
                raw = re.sub(r"```$", "", raw).strip()

                try:
                    obj = json.loads(raw)
                except Exception:
                    obj = {
                        "headline": f"{section_title} — AI narrative",
                        "what": raw,
                        "so_what": "",
                        "now_what": "",
                        "performance_rows": [],
                        "recommendations": []
                    }

                _render_exec_cards(
                    obj.get("headline", f"{section_title} — Summary"),
                    obj.get("what",""),
                    obj.get("so_what",""),
                    obj.get("now_what","")
                )

                tabs = st.tabs(["Performance & Use Cases", "Recommendations", "Risks"])
                with tabs[0]:
                    _render_performance_table(obj.get("performance_rows", []))
                with tabs[1]:
                    _render_recommendations(obj.get("recommendations", []))
                with tabs[2]:
                    risks = obj.get("risks", [])
                    if risks:
                        for r in risks:
                            st.markdown(f"- {_markdown_escape(r)}")
                    else:
                        st.info("No explicit risks flagged.")

                _download_narrative(obj, filename_prefix=f"{section_title.lower().replace(' ','_')}_{'exec' if mode.startswith('Exec') else 'analyst'}")

            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
                st.json(payload)
        else:
            # Deterministic, structured fallback for both audiences
            headline = f"{section_title} — Summary"
            # Best-effort KPIs pull from payload keys
            kpi_pairs = []
            if "metrics" in payload and isinstance(payload["metrics"], dict):
                for k, v in payload["metrics"].items():
                    kpi_pairs.append((k, v))
            cols = st.columns(min(3, max(1, len(kpi_pairs))))
            for i, (k, v) in enumerate(kpi_pairs[:3]):
                with cols[i]:
                    st.markdown(f'<div class="kpi"><div class="small">{_markdown_escape(k)}</div><h3>{_markdown_escape(v)}</h3></div>', unsafe_allow_html=True)

            what = "Key results prepared deterministically from the current payload."
            so_what = "Impacts focus on operational cost, throughput, and risk exposure given the observed metrics."
            now_what = "Consider tightening policies on high-cost segments, validate anomalies, and pilot staffing adjustments aligned to forecast."

            _render_exec_cards(headline, what, so_what, now_what)

            # Build naive performance rows if we find common fields
            perf_rows = []
            if "rows_used" in payload:
                perf_rows.append({"metric":"Rows used","value":payload["rows_used"],"use_case":"Model context","owner":"Data Eng"})
            if "metrics" in payload and "MAE" in payload["metrics"]:
                perf_rows.append({"metric":"MAE","value":payload["metrics"]["MAE"],"use_case":"Billing prediction error","owner":"Data Science"})
            if "metrics" in payload and "R2" in payload["metrics"]:
                perf_rows.append({"metric":"R²","value":payload["metrics"]["R2"],"use_case":"Model fit","owner":"Data Science"})
            tabs = st.tabs(["Performance & Use Cases", "Recommendations"])
            with tabs[0]:
                _render_performance_table(perf_rows)
            with tabs[1]:
                _render_recommendations([
                    {"action":"Validate outliers with clinical ops","owner":"Ops","sla":"7 days","rationale":"Anomalies may signal coding or process drift","confidence":0.7},
                    {"action":"Pilot budget guardrails in optimizer","owner":"Finance","sla":"14 days","rationale":"Bound cost exposure while we improve prediction","confidence":0.8},
                    {"action":"Implement RN ratio alerting","owner":"Nursing","sla":"21 days","rationale":"Smooth staffing spikes based on forecast","confidence":0.75},
                ])
            # Minimal JSON/MD export
            obj = {
                "headline": headline, "what": what, "so_what": so_what, "now_what": now_what,
                "performance_rows": perf_rows,
                "recommendations": [
                    {"action":"Validate outliers with clinical ops","owner":"Ops","sla":"7 days","rationale":"Anomalies may signal coding or process drift","confidence":0.7},
                    {"action":"Pilot budget guardrails in optimizer","owner":"Finance","sla":"14 days","rationale":"Bound cost exposure while we improve prediction","confidence":0.8},
                    {"action":"Implement RN ratio alerting","owner":"Nursing","sla":"21 days","rationale":"Smooth staffing spikes based on forecast","confidence":0.75},
                ]
            }
            _download_narrative(obj, filename_prefix=f"{section_title.lower().replace(' ','_')}_{'exec' if mode.startswith('Exec') else 'analyst'}")

# ---------------- NAV ----------------
tabs = st.tabs(["💵 Price Prediction", "🧮 Case Mix Optimizer", "👩‍⚕️ Staffing Optimizer", "📈 Trends"])

# ======================================================================================
# 1) PRICE PREDICTION (scikit-learn)
# ======================================================================================
with tabs[0]:
    st.subheader("💵 Price Prediction — Expected Billing per Case")
    st.caption("Train a quick model to predict billing, see accuracy & feature drivers, and use predictions downstream in optimization.")

    # Features for regression
    num_cols = [c for c in ["age","length_of_stay","dow","month"] if c in df.columns]
    cat_cols = [c for c in ["gender","insurer","hospital","doctor","condition","admission_type","medication"] if c in df.columns]
    use_cols = num_cols + cat_cols + (["billing_amount"] if "billing_amount" in df.columns else [])

    if "billing_amount" not in df.columns:
        st.error("The dataset is missing 'billing_amount'. Cannot train the price prediction model.")
    else:
        data = df.dropna(subset=["billing_amount"])[use_cols].copy()

        MIN_ROWS_FOR_MODEL = 150  # unified threshold (more robust for high-cardinality OHE)
        if len(data) < MIN_ROWS_FOR_MODEL:
            st.info(f"Not enough rows to train a robust model (need ≈{MIN_ROWS_FOR_MODEL}+).")
        else:
            X = data.drop(columns=["billing_amount"])
            y = data["billing_amount"].astype(float)

            # Robust preprocessing: impute then scale / encode
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
            ])
            pre = ColumnTransformer([
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ], remainder="drop")

            model = Pipeline([
                ("pre", pre),
                ("rf", RandomForestRegressor(
                    n_estimators=250, random_state=42, n_jobs=-1, max_depth=None, min_samples_leaf=3
                ))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-9, None))) * 100
            r2 = r2_score(y_test, y_pred)

            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="kpi"><div class="small">MAE</div><h3>${mae:,.0f}</h3></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><div class="small">MAPE</div><h3>{mape:.1f}%</h3></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><div class="small">R²</div><h3>{r2:.2f}</h3></div>', unsafe_allow_html=True)

            # Scatter: Actual vs Predicted
            plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            fig_sc = px.scatter(plot_df, x="Actual", y="Predicted", title="Actual vs Predicted Billing",
                                trendline="ols", opacity=0.5)
            st.plotly_chart(fig_sc, use_container_width=True)

            # Permutation importance with safeguards (can be expensive)
            st.markdown("#### Top Drivers (Permutation Importance)")
            can_do_perm = True
            # Estimate expanded feature size; if too large, skip
            try:
                ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
                cat_size = int(sum(len(df[c].astype("category").cat.categories) for c in cat_cols)) if cat_cols else 0
                approx_features = len(num_cols) + cat_size
                if approx_features > 3000 or len(X_test) > 20000:
                    can_do_perm = False
            except Exception:
                pass

            do_perm = st.toggle("Compute permutation importance (may be slow)", value=can_do_perm)
            if do_perm:
                try:
                    with st.spinner("Computing permutation importance..."):
                        # Sample test set for speed
                        sample_size = min(1000, len(X_test))
                        Xs = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
                        ys = y_test.loc[Xs.index]
                        imp = permutation_importance(model, Xs, ys, n_repeats=3, random_state=42, n_jobs=-1)

                        # Try robust feature name extraction
                        feature_names: List[str] = []
                        try:
                            # sklearn >= 1.0 supports get_feature_names_out on the full preprocessor
                            feature_names = model.named_steps["pre"].get_feature_names_out().tolist()
                        except Exception:
                            # Fallback: numeric names + OHE names
                            feature_names = []
                            # numeric
                            feature_names += num_cols
                            # categorical expanded
                            try:
                                ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
                                cat_names = ohe.get_feature_names_out(cat_cols).tolist()
                            except Exception:
                                cat_names = [f"{c}_encoded" for c in cat_cols]
                            feature_names += cat_names

                        k = min(len(feature_names), len(imp.importances_mean))
                        importances = pd.DataFrame({
                            "feature": feature_names[:k],
                            "importance": imp.importances_mean[:k]
                        }).sort_values("importance", ascending=False).head(20)

                        fig_imp = px.bar(importances, x="feature", y="importance", title="Top Features (Permutation Importance)", height=420)
                        fig_imp.update_layout(xaxis_tickangle=-35)
                        st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.info(f"Permutation importance unavailable; continuing. ({e})")
            else:
                st.caption("Skipped permutation importance to keep things snappy.")

            # Save predictions back to df for optimization (robust to missing columns)
            try:
                df["_predicted_billing"] = safe_predict(model, df[X.columns], list(X.columns))
            except Exception:
                # Align any current columns that intersect, fill the rest with NaN (handled by imputers)
                intersect = [c for c in X.columns if c in df.columns]
                temp = df[intersect].copy()
                for missing in [c for c in X.columns if c not in temp.columns]:
                    temp[missing] = np.nan
                df["_predicted_billing"] = safe_predict(model, temp[X.columns], list(X.columns))

            # AI analyst
            payload = {
                "rows_used": int(len(data)),
                "metrics": {"MAE": float(mae), "MAPE%": float(mape), "R2": float(r2)},
                "sample_predictions": plot_df.sample(min(10, len(plot_df)), random_state=1).round(2).to_dict("records")
            }
            st.markdown("---")
            ai_write("Price Prediction", payload)

# ======================================================================================
# 2) CASE MIX OPTIMIZER (MILP using predicted prices)
# ======================================================================================
with tabs[1]:
    st.subheader("🧮 Case Mix Optimizer — Minimize Predicted Billing under Policy Limits")
    st.caption("Uses predicted billing (from the Price Prediction module) to choose a feasible elective mix that reduces cost exposure.")

    # Working set
    cols = ["_predicted_billing","billing_amount","is_weekend","is_longstay","anomaly_flag",
            "admit_date","insurer","hospital","doctor","condition","length_of_stay","gender"]
    base = df.dropna(subset=["billing_amount"]).copy()
    if "_predicted_billing" not in base.columns:
        base["_predicted_billing"] = base["billing_amount"].astype(float)

    # Only keep columns that exist
    cols_present = [c for c in cols if c in base.columns]
    work_all = base[cols_present].dropna(subset=["_predicted_billing", "billing_amount"])
    if len(work_all) == 0:
        st.error("No records available for optimization. Train the model first or check your data.")
    else:
        sample_n = st.slider("Sampling size (optimization set)", 100, min(2000, len(work_all)), min(250, len(work_all)), step=50, key="opt_samp")
        work = work_all.sample(n=min(sample_n, len(work_all)), random_state=42).reset_index(drop=True)

        # Policy controls
        c1, c2, c3, c4 = st.columns(4)
        target_cases = c1.number_input("Target number of cases", min_value=20, max_value=min(1000, len(work)), value=min(120, len(work)))
        max_weekend_pct = c2.slider("Max weekend %", 0, 100, 20, 5)
        max_longstay_pct = c3.slider("Max long-stay %", 0, 100, 12, 2)
        max_anomaly_pct = c4.slider("Max anomaly %", 0, 100, 6, 1)

        # Optional budget ceiling (soft; can be toggled on)
        c5, c6 = st.columns([1,3])
        use_budget = c5.checkbox("Add budget ceiling", value=False)
        budget_value = c6.number_input(
            "Budget ceiling ($, applies if checked)",
            min_value=0.0,
            value=float(work["_predicted_billing"].sum()*0.6)
        )

        # MILP: minimize predicted total cost subject to constraints
        model = LpProblem("CaseMix_MinPredictedCost", LpMinimize)
        x = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in range(len(work))]
        cost_vec = work["_predicted_billing"].values

        model += lpSum(cost_vec[i] * x[i] for i in range(len(work)))
        model += lpSum(x) == int(target_cases)
        if "is_weekend" in work.columns:
            model += lpSum(work["is_weekend"].values[i] * x[i] for i in range(len(work))) <= (max_weekend_pct/100.0) * target_cases
        if "is_longstay" in work.columns:
            model += lpSum(work["is_longstay"].values[i] * x[i] for i in range(len(work))) <= (max_longstay_pct/100.0) * target_cases
        if "anomaly_flag" in work.columns:
            model += lpSum(work["anomaly_flag"].values[i] * x[i] for i in range(len(work))) <= (max_anomaly_pct/100.0) * target_cases
        if use_budget:
            model += lpSum(cost_vec[i] * x[i] for i in range(len(work))) <= budget_value

        _ = model.solve(PULP_CBC_CMD(msg=False))
        feasible = (LpStatus[model.status] == "Optimal")

        if not feasible:
            st.error("🚫 Infeasible optimization — loosen one or more constraints.")
        else:
            work["selected"] = [int(v.value()) for v in x]
            chosen = work[work["selected"] == 1].copy()

            # KPIs & ROI view (compare predicted vs actual baseline)
            pred_total = float(chosen["_predicted_billing"].sum())
            actual_total = float(chosen["billing_amount"].sum())
            naive_topn = work["_predicted_billing"].sort_values().head(int(target_cases)).sum()
            savings_vs_naive = float(naive_topn - pred_total)

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="kpi"><div class="small">Selected Cases</div><h3>{len(chosen):,}</h3></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><div class="small">Predicted Total Billing</div><h3>${pred_total:,.0f}</h3></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><div class="small">Actual Total (selected)</div><h3>${actual_total:,.0f}</h3></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi"><div class="small">Δ vs naive (pred-topN)</div><h3>${savings_vs_naive:,.0f}</h3></div>', unsafe_allow_html=True)

            # Composition charts
            cA, cB = st.columns(2)
            with cA:
                if "insurer" in chosen.columns:
                    by_insurer = chosen.groupby("insurer")["_predicted_billing"].sum().sort_values(ascending=False).head(12).reset_index()
                    fig = px.bar(by_insurer, x="insurer", y="_predicted_billing", title="Predicted Cost by Insurer")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No 'insurer' column available for composition chart.")
                    by_insurer = pd.DataFrame()
            with cB:
                if "condition" in chosen.columns:
                    by_cond = chosen.groupby("condition")["_predicted_billing"].sum().sort_values(ascending=False).head(12).reset_index()
                    fig = px.bar(by_cond, x="condition", y="_predicted_billing", title="Predicted Cost by Condition")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No 'condition' column available for composition chart.")
                    by_cond = pd.DataFrame()

            st.markdown("#### Selected Cases (sample)")
            show_cols = [c for c in ["admit_date","hospital","doctor","insurer","condition","_predicted_billing","billing_amount",
                                     "length_of_stay","is_weekend","is_longstay","anomaly_flag"] if c in chosen.columns]
            st.dataframe(chosen[show_cols].sort_values("_predicted_billing", ascending=False).head(250),
                         use_container_width=True, hide_index=True)

            # Download
            st.download_button("⬇️ Download selected cases (CSV)",
                               data=chosen.to_csv(index=False),
                               file_name="selected_cases.csv",
                               mime="text/csv")

            # AI Analyst
            payload = {
                "selected_cases": int(len(chosen)),
                "predicted_total_billing": float(pred_total),
                "actual_total_billing": float(actual_total),
                "policy_limits": {"weekend_pct": max_weekend_pct, "longstay_pct": max_longstay_pct, "anomaly_pct": max_anomaly_pct},
                "composition": {
                    "by_insurer": by_insurer.to_dict("records") if not by_insurer.empty else None,
                    "by_condition": by_cond.to_dict("records") if not by_cond.empty else None
                },
                "budget_used": float(pred_total) if use_budget else None,
                "budget_ceiling": float(budget_value) if use_budget else None
            }
            st.markdown("---")
            ai_write("Case Mix Optimizer", payload)

# ======================================================================================
# 3) STAFFING OPTIMIZER (forecast -> integer RNs per shift)
# ======================================================================================
with tabs[2]:
    st.subheader("👩‍⚕️ Staffing Optimizer — RN Targets from Admissions Forecast")
    st.caption("Minimize staffing cost while covering forecast workload; three integer shift variables per day (Day/Evening/Night).")

    # Forecast horizon (relative to inferred frequency)
    horizon = st.slider("Forecast horizon (periods)", 7, 28, 14, step=7, key="staff_h")
    fc = admissions_forecast(df, horizon=horizon)
    if fc.empty:
        st.info("Not enough admissions history to forecast.")
    else:
        # Display forecast
        fig = go.Figure()
        hist = build_timeseries(df, "intake")
        if not hist.empty:
            fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="History", mode="lines"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast", mode="lines"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=50))
        st.plotly_chart(fig, use_container_width=True)

        # Assumptions
        c1, c2, c3, c4 = st.columns(4)
        ratio = c1.number_input("Patients per RN per shift", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
        cost_day = c2.number_input("RN cost (Day)", min_value=100.0, max_value=2000.0, value=650.0, step=25.0)
        cost_eve = c3.number_input("RN cost (Evening)", min_value=100.0, max_value=2000.0, value=700.0, step=25.0)
        cost_night = c4.number_input("RN cost (Night)", min_value=100.0, max_value=2000.0, value=750.0, step=25.0)

        need = np.ceil(fc["yhat"].values / ratio).astype(int)

        RN_day = [LpVariable(f"RN_day_{d}", lowBound=0, cat=LpInteger) for d in range(horizon)]
        RN_eve = [LpVariable(f"RN_eve_{d}", lowBound=0, cat=LpInteger) for d in range(horizon)]
        RN_nig = [LpVariable(f"RN_nig_{d}", lowBound=0, cat=LpInteger) for d in range(horizon)]

        staff_model = LpProblem("Staffing_Optimization", LpMinimize)
        staff_model += lpSum(cost_day*RN_day[d] + cost_eve*RN_eve[d] + cost_night*RN_nig[d] for d in range(horizon))
        for d in range(horizon):
            staff_model += RN_day[d] + RN_eve[d] + RN_nig[d] >= int(need[d])

        _ = staff_model.solve(PULP_CBC_CMD(msg=False))
        feasible = (LpStatus[staff_model.status] == "Optimal")
        if not feasible:
            st.error("🚫 Staffing optimization infeasible with current parameters.")
        else:
            sol = pd.DataFrame({
                "Date": pd.to_datetime(fc["ds"]).date,
                "RN_Day": [int(RN_day[d].value()) for d in range(horizon)],
                "RN_Evening": [int(RN_eve[d].value()) for d in range(horizon)],
                "RN_Night": [int(RN_nig[d].value()) for d in range(horizon)],
            })
            sol["Total_RN"] = sol[["RN_Day","RN_Evening","RN_Night"]].sum(axis=1)
            sol["Daily Cost"] = sol["RN_Day"]*cost_day + sol["RN_Evening"]*cost_eve + sol["RN_Night"]*cost_night

            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="kpi"><div class="small">Avg RN/day</div><h3>{sol["Total_RN"].mean():.1f}</h3></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><div class="small">Total Cost ({horizon} periods)</div><h3>${sol["Daily Cost"].sum():,.0f}</h3></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><div class="small">Patients per RN</div><h3>{ratio:.1f}</h3></div>', unsafe_allow_html=True)

            st.markdown("#### RN Targets by Day/Period")
            st.dataframe(sol, use_container_width=True, hide_index=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=sol["Date"], y=sol["RN_Day"], name="Day"))
            fig2.add_trace(go.Bar(x=sol["Date"], y=sol["RN_Evening"], name="Evening"))
            fig2.add_trace(go.Bar(x=sol["Date"], y=sol["RN_Night"], name="Night"))
            fig2.update_layout(barmode="stack", height=380, title="RN Staffing Plan (stacked by shift)")
            st.plotly_chart(fig2, use_container_width=True)

            # AI
            payload = {
                "horizon_periods": horizon,
                "admissions_forecast_avg": float(np.mean(fc["yhat"])),
                "patients_per_RN": float(ratio),
                "total_cost": float(sol["Daily Cost"].sum()),
                "avg_RN_per_period": float(sol["Total_RN"].mean()),
            }
            st.markdown("---")
            ai_write("Staffing Optimizer", payload)

# ======================================================================================
# 4) TRENDS
# ======================================================================================
with tabs[3]:
    st.subheader("📈 Trends & Explainability")

    # Choose frequency based on data; present that to user to avoid confusion
    inferred_freq, _ = infer_ts_granularity(df["admit_date"].sort_values() if "admit_date" in df else pd.Series([], dtype="datetime64[ns]"))
    st.caption(f"Time series frequency inferred: **{inferred_freq}**")

    c1, c2 = st.columns(2)
    with c1:
        ts_bill = build_timeseries(df, "billing_amount")  # inferred freq
        if ts_bill.empty:
            st.info("No billing time series available.")
        else:
            fig = px.line(ts_bill, x="ds", y="y", title=f"Billing — {inferred_freq}")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "length_of_stay" in df.columns:
            ts_los = build_timeseries(df, "length_of_stay")
            if ts_los.empty:
                st.info("No LOS time series available.")
            else:
                fig = px.line(ts_los, x="ds", y="y", title=f"Avg LOS — {inferred_freq}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LOS field available.")

    st.markdown("#### Current Anomaly Mix")
    if "insurer" in df.columns and df["insurer"].notna().any():
        mix = df.groupby("insurer", dropna=False).agg(
            total_cost=("billing_amount","sum"),
            n=("billing_amount","count"),
            anomaly_rate=("anomaly_flag","mean")
        ).reset_index().sort_values("total_cost", ascending=False).head(15)
        fig = px.bar(mix, x="insurer", y="total_cost", color="anomaly_rate",
                     title="Top Insurers by Cost (color = anomaly rate)", labels={"total_cost":"Total Billing"})
        st.plotly_chart(fig, use_container_width=True)
    elif "hospital" in df.columns and df["hospital"].notna().any():
        st.info("No 'insurer' column usable; showing anomaly mix by hospital instead.")
        mix = df.groupby("hospital", dropna=False).agg(
            total_cost=("billing_amount","sum"),
            n=("billing_amount","count"),
            anomaly_rate=("anomaly_flag","mean")
        ).reset_index().sort_values("total_cost", ascending=False).head(15)
        fig = px.bar(mix, x="hospital", y="total_cost", color="anomaly_rate",
                     title="Top Hospitals by Cost (color = anomaly rate)", labels={"total_cost":"Total Billing"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        mix = pd.DataFrame()
        st.info("No 'insurer' or 'hospital' group available; skipping anomaly mix chart.")

    payload = {
        "billing_weekly_tail": ts_bill.tail(12).to_dict("records") if not ts_bill.empty else None,
        "los_weekly_tail": (build_timeseries(df, 'length_of_stay').tail(12).to_dict("records") if 'length_of_stay' in df.columns else None),
        "anomaly_by_group": (mix.to_dict("records") if not mix.empty else None),
    }
    st.markdown("---")
    ai_write("Trends", payload)

# ======================= FOOTER =======================
st.markdown("---")
st.caption("Built for healthcare operations — simple deployment, big decisions. © 2025")
