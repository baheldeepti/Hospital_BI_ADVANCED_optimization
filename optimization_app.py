# app.py
# Hospital Optimization Studio — Feedback-Integrated Build (Fixed OHE param)
# Modules: Price Prediction • Case Mix Optimizer • Staffing • Trends
# No Gurobi; portable stack; OpenAI optional

import os, json, textwrap, re, math, warnings
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

# Import ONLY needed tsa pieces; avoid statsmodels.api to dodge SciPy _lazywhere path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize, LpInteger, LpBinary,
    LpStatus, PULP_CBC_CMD
)

warnings.filterwarnings("ignore")

# ------------------------ CONSTANTS ------------------------
RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"

MIN_ROWS_FOR_MODEL = 150
MAX_PERM_FEATURES = 3000
MAX_PERM_TEST_ROWS = 1000
ANOMALY_MIN_ROWS = 200
ANOMALY_CONTAM = 0.03
PI_EPS_LOS_DAYS = 0.25   # 6 hours tolerance for LOS jitter
DEFAULT_OPT_SAMPLE = 250
MAX_OPT_SAMPLE = 2000

THEME_CSS = """
<style>
:root{
  --ink:#0b2740; --pri:#0F4C81; --teal:#159E99; --sub:#5b6b7a; --bg:#f7f9fb;
}
html, body, [class^="css"]  {background-color: var(--bg);}
.block-container{max-width:1500px;padding-top:12px}
h1,h2,h3{font-weight:700;color:#0b2740}
.stTabs [data-baseweb="tab-list"] { gap: 6px }
.stTabs [data-baseweb="tab"]{
  background: white; padding: 8px 14px; border-radius: 10px; border: 1px solid #e8eef5;
}
.stTabs [aria-selected="true"]{
  background: #e8f2ff; border-color:#c8defc;
}
a {color:#159E99}
.stButton>button{background:#0F4C81;color:#fff;border-radius:10px;border:0}
.badge{display:inline-block;padding:.25rem .55rem;border-radius:.5rem;background:#eef3f7;color:#0b2740;margin-right:.3rem}
.small{color:#5b6b7a;font-size:0.92rem}
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
"""

# ------------------------ PAGE SETUP ------------------------
st.set_page_config(page_title="Hospital Optimization Studio", layout="wide", page_icon="🏥")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.title("🏥 Hospital Optimization Studio")
st.caption("Price Prediction • Case Mix Optimization • Staffing Targets • Trends • AI Narrative")

# ------------------------ HELPERS ------------------------
def _markdown_escape(text: str) -> str:
    return re.sub(r'([*_`])', r'\\\1', str(text))

def ensure_columns(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        for m in missing:
            frame[m] = np.nan
    return frame[cols]

def safe_predict(pipeline: Pipeline, X_like: pd.DataFrame, cols_needed: Sequence[str]) -> np.ndarray:
    X_aligned = ensure_columns(X_like.copy(), cols_needed)
    return pipeline.predict(X_aligned)

def _most_common_gap_days(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2: return 1.0
    d = np.diff(np.sort(idx.values).astype("datetime64[D]").astype("int64"))
    if len(d) == 0: return 1.0
    vals, counts = np.unique(d, return_counts=True)
    return float(vals[np.argmax(counts)])

def infer_ts_granularity(idx: pd.DatetimeIndex) -> Tuple[str, Optional[int]]:
    """
    More robust frequency inference:
    - If daily-ish with frequent 2–3 day gaps clustered around weekends => 'B' (business day)
    - Else if median gap <= 1.5 => 'D'
    - Else if median gap <= 10 => 'W'
    - Else 'M'
    Returns (freq, seasonal_periods)
    """
    if len(idx) < 3:
        return "D", None

    idx_sorted = pd.DatetimeIndex(sorted(idx))
    diffs_days = np.diff(idx_sorted.values).astype("timedelta64[D]").astype(float)
    if len(diffs_days) == 0:
        return "D", 7

    med_gap = float(np.median(diffs_days))
    gap3 = np.mean(np.isclose(diffs_days, 3.0)) if len(diffs_days) else 0.0
    gap2 = np.mean(np.isclose(diffs_days, 2.0)) if len(diffs_days) else 0.0
    big_gaps = np.mean(diffs_days > 7.5)

    # Weekend pattern heuristic: lots of 2–3 day gaps, few big gaps, and Fridays represented
    fri_ratio = np.mean(idx_sorted.weekday == 4) if len(idx_sorted) else 0.0
    if med_gap <= 1.6 and (gap2 + gap3) >= 0.35 and big_gaps <= 0.1 and fri_ratio >= 0.10:
        return "B", 5 if len(idx_sorted) >= 30 else None

    if med_gap <= 1.5:
        return "D", 7
    elif med_gap <= 10:
        return "W", 52 if len(idx_sorted) >= 52 else None
    else:
        return "M", 12 if len(idx_sorted) >= 24 else None

def build_timeseries(data: pd.DataFrame, metric: str, freq: Optional[str] = None) -> pd.DataFrame:
    if "admit_date" not in data.columns:
        return pd.DataFrame(columns=["ds","y"])
    idx_df = data.set_index("admit_date").sort_index()
    if freq is None:
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
    ts = build_timeseries(data, "intake")
    if ts.empty:
        return pd.DataFrame(columns=["ds","yhat"])
    s = ts.set_index("ds")["y"]
    inferred_freq, seasonal_periods = infer_ts_granularity(s.index)
    s = s.asfreq(inferred_freq).ffill()

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

def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
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
        if k in df.columns: df.rename(columns={k: v}, inplace=True)

    if "admit_date" in df.columns:
        df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
    if "discharge_date" in df.columns:
        df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    for c in ["billing_amount", "length_of_stay", "age"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # LOS compute + tolerance
    if "length_of_stay" in df.columns:
        pass
    elif {"admit_date","discharge_date"}.issubset(df.columns):
        sec = (df["discharge_date"] - df["admit_date"]).dt.total_seconds()
        df["length_of_stay"] = sec / 86400.0
    else:
        df["length_of_stay"] = np.nan

    # Apply LOS jitter guard
    if "length_of_stay" in df.columns:
        los = pd.to_numeric(df["length_of_stay"], errors="coerce")
        small_neg = (los < 0) & (los >= -PI_EPS_LOS_DAYS)
        los[small_neg] = 0.0
        los[los < -PI_EPS_LOS_DAYS] = np.nan
        df["length_of_stay"] = los

    # Basic hygiene
    req_for_model = {"admit_date", "billing_amount"}
    if not req_for_model.issubset(df.columns):
        st.warning("Dataset missing required columns for modeling: "
                   + ", ".join(sorted(req_for_model - set(df.columns))))
    df = df.dropna(subset=["admit_date", "billing_amount"])

    # Feature flags
    df["dow"] = df["admit_date"].dt.weekday
    df["month"] = df["admit_date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    for c in ["gender","insurer","hospital","doctor","condition","medication","admission_type"]:
        if c in df.columns: df[c] = df[c].fillna("Unknown")

    # Lightweight anomaly flag
    try:
        if "billing_amount" in df.columns and len(df) >= ANOMALY_MIN_ROWS:
            iso = IsolationForest(contamination=ANOMALY_CONTAM, random_state=42)
            df["anomaly_flag"] = (iso.fit_predict(df[["billing_amount"]]) == -1).astype(int)
        else:
            df["anomaly_flag"] = 0
    except Exception:
        df["anomaly_flag"] = 0

    return df

# Back/forward compatible OneHotEncoder factory
def make_ohe():
    """
    sklearn >=1.4/1.5 removed 'sparse' in favor of 'sparse_output'.
    Try new param first, then fall back for older versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# ------------------------ DATA LOAD ------------------------
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    try:
        local = "/mnt/data/modified_healthcare_dataset.csv"
        if os.path.exists(local):
            df = pd.read_csv(local)
        else:
            df = pd.read_csv(RAW_URL)
    except Exception:
        df = pd.read_csv(RAW_URL)
    return validate_and_normalize(df)

df = load_default()

with st.expander("📁 Data source (optional override)"):
    up = st.file_uploader("Upload CSV to override default", type=["csv"])
    if up is not None:
        try:
            up_df = pd.read_csv(up)
            df = validate_and_normalize(up_df)
            st.success(f"Loaded {len(df):,} rows from upload.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# ------------------------ AI (optional) ------------------------
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

def ai_write(section_title: str, payload: dict):
    client = get_openai_client()
    colL, colR = st.columns([1,2])
    use_ai = colL.checkbox(f"Use AI for {section_title}", value=(client is not None), key=f"ai_{section_title}")
    mode = colR.radio("Audience", ["Executive Summary", "Analyst Deep-Dive"], horizontal=True, key=f"mode_{section_title}")

    st.markdown(f"<div class='pill'>Section: {section_title}</div>", unsafe_allow_html=True)

    if use_ai and client:
        schema_hint = {
            "headline": "string",
            "what": "string",
            "so_what": "string",
            "now_what": "string",
            "performance_rows": [{"metric":"string","value":"string/number","use_case":"string","owner":"string"}],
            "recommendations": [{"action":"string","owner":"string","sla":"string","rationale":"string","confidence":"0.0-1.0"}],
            "risks": ["string"]
        }
        prompt = textwrap.dedent(f"""
        Respond with STRICT JSON only (no prose), per this schema:
        {json.dumps(schema_hint, indent=2)}
        Audience: {"executive" if mode.startswith("Executive") else "analyst"}
        Keep total words ~140–220. Use only this JSON input:
        {json.dumps(payload, default=str)[:6000]}
        """).strip()
        try:
            rsp = client.chat.completions.create(
                model=(st.secrets.get("PREFERRED_OPENAI_MODEL") or "gpt-4o-mini"),
                messages=[
                    {"role":"system","content":"Respond with STRICT JSON that conforms to the provided schema. No extra text."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2
            )
            raw = rsp.choices[0].message.content.strip()
            raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"```$", "", raw).strip()
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"headline": f"{section_title} — AI narrative", "what": raw, "so_what": "", "now_what": "",
                       "performance_rows": [], "recommendations": [], "risks":[]}
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
            obj = {"headline": f"{section_title} — Narrative", "what": "", "so_what": "", "now_what": "",
                   "performance_rows": [], "recommendations": [], "risks":[]}
    else:
        # deterministic fallback
        obj = {
            "headline": f"{section_title} — Summary",
            "what": "Key results computed deterministically from the current run.",
            "so_what": "Operational impact on cost, throughput, and risk.",
            "now_what": "Tighten guardrails, validate anomalies, and pilot staffing tweaks.",
            "performance_rows": [],
            "recommendations": [
                {"action":"Validate outliers with clinical ops","owner":"Ops","sla":"7 days","rationale":"Possible coding drift","confidence":"0.7"},
                {"action":"Pilot budget guardrails in optimizer","owner":"Finance","sla":"14 days","rationale":"Bound exposure","confidence":"0.8"},
            ],
            "risks":[]
        }

    # render
    st.markdown(f"""
    <div class="card"><h4>Headline</h4><div>{_markdown_escape(obj.get('headline',''))}</div></div>
    <div class="card"><h4>What happened</h4><div>{_markdown_escape(obj.get('what',''))}</div></div>
    <div class="card"><h4>Why it matters</h4><div>{_markdown_escape(obj.get('so_what',''))}</div></div>
    <div class="card"><h4>What we should do</h4><div>{_markdown_escape(obj.get('now_what',''))}</div></div>
    """, unsafe_allow_html=True)

    pr = obj.get("performance_rows", [])
    if pr:
        dfp = pd.DataFrame(pr)
        st.markdown("**Performance & Use Cases**")
        st.dataframe(dfp, use_container_width=True, hide_index=True)

    recs = obj.get("recommendations", [])
    if recs:
        dfr = pd.DataFrame(recs)
        if "confidence" in dfr.columns:
            def fmtc(x):
                try: return f"{round(float(x)*100):.0f}%"
                except: return ""
            dfr["confidence"] = dfr["confidence"].apply(fmtc)
        st.markdown("**Recommendations (Owners & SLAs)**")
        st.dataframe(dfr, use_container_width=True, hide_index=True)

    # download
    st.download_button("⬇️ Download narrative (JSON)", data=json.dumps(obj, indent=2),
                       file_name=f"{section_title.lower().replace(' ','_')}.json", mime="application/json")

# ------------------------ NAV ------------------------
tabs = st.tabs(["💵 Price Prediction", "🧮 Case Mix Optimizer", "👩‍⚕️ Staffing Optimizer", "📈 Trends"])

# ===================== 1) PRICE PREDICTION =====================
with tabs[0]:
    st.subheader("💵 Price Prediction — Expected Billing per Case")
    st.caption("Train a quick model, check accuracy & drivers, and feed predictions to the optimizer.")

    num_cols = [c for c in ["age","length_of_stay","dow","month"] if c in df.columns]
    cat_cols = [c for c in ["gender","insurer","hospital","doctor","condition","admission_type","medication"] if c in df.columns]

    if "billing_amount" not in df.columns:
        st.error("Dataset is missing 'billing_amount'.")
    else:
        data = df.dropna(subset=["billing_amount"])[num_cols + cat_cols + ["billing_amount"]].copy()

        if len(data) < MIN_ROWS_FOR_MODEL:
            st.info(f"Not enough rows to train robustly (need ≈{MIN_ROWS_FOR_MODEL}+).")
        else:
            X = data.drop(columns=["billing_amount"])
            y = data["billing_amount"].astype(float)

            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])
            cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("ohe", make_ohe())])
            pre = ColumnTransformer([("num", num_pipe, num_cols),
                                     ("cat", cat_pipe, cat_cols)], remainder="drop")

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

            # Scatter: Actual vs Predicted + lightweight OLS line (no statsmodels.api)
            plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            fig_sc = px.scatter(plot_df, x="Actual", y="Predicted", title="Actual vs Predicted Billing", opacity=0.5)
            try:
                x = plot_df["Actual"].values
                yhat = plot_df["Predicted"].values
                b1, b0 = np.polyfit(x, yhat, 1)  # yhat ≈ b1*x + b0
                xs = np.linspace(x.min(), x.max(), 100)
                ys = b1*xs + b0
                fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="OLS (np.polyfit)"))
            except Exception:
                pass
            st.plotly_chart(fig_sc, use_container_width=True)

            # Permutation importance (guarded)
            st.markdown("#### Top Drivers (Permutation Importance)")
            can_do_perm = False
            try:
                ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
                cat_size = int(sum(len(df[c].astype("category").cat.categories) for c in cat_cols)) if cat_cols else 0
                approx_features = len(num_cols) + cat_size
                can_do_perm = (approx_features <= MAX_PERM_FEATURES and len(X_test) >= 50)
            except Exception:
                can_do_perm = False

            do_perm = st.toggle("Compute permutation importance (may be slow)", value=can_do_perm)
            if do_perm:
                try:
                    with st.spinner("Computing permutation importance..."):
                        sample_size = min(MAX_PERM_TEST_ROWS, len(X_test))
                        Xs = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
                        ys = y_test.loc[Xs.index]
                        imp = permutation_importance(model, Xs, ys, n_repeats=3, random_state=42, n_jobs=-1)

                        # Robust feature names
                        try:
                            feature_names = model.named_steps["pre"].get_feature_names_out().tolist()
                        except Exception:
                            feature_names = []
                            feature_names += num_cols
                            try:
                                ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
                                feature_names += ohe.get_feature_names_out(cat_cols).tolist()
                            except Exception:
                                feature_names += [f"{c}_encoded" for c in cat_cols]

                        k = min(len(feature_names), len(imp.importances_mean))
                        importances = pd.DataFrame({
                            "feature": feature_names[:k],
                            "importance": imp.importances_mean[:k]
                        }).sort_values("importance", ascending=False).head(20)

                        fig_imp = px.bar(importances, x="feature", y="importance",
                                         title="Top Features (Permutation Importance)", height=420)
                        fig_imp.update_layout(xaxis_tickangle=-35)
                        st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.info(f"Permutation importance unavailable; continuing. ({e})")
            else:
                st.caption("Skipped permutation importance to keep things snappy.")

            # Persist predictions for optimizer
            try:
                df["_predicted_billing"] = safe_predict(model, df[X.columns], list(X.columns))
            except Exception:
                intersect = [c for c in X.columns if c in df.columns]
                temp = df[intersect].copy()
                for missing in [c for c in X.columns if c not in temp.columns]:
                    temp[missing] = np.nan
                temp = temp[[*X.columns]]
                df["_predicted_billing"] = safe_predict(model, temp, list(X.columns))

            payload = {
                "rows_used": int(len(data)),
                "metrics": {"MAE": float(mae), "MAPE%": float(mape), "R2": float(r2)},
                "sample_predictions": plot_df.sample(min(10, len(plot_df)), random_state=1).round(2).to_dict("records")
            }
            st.markdown("---")
            ai_write("Price Prediction", payload)

# ===================== 2) CASE MIX OPTIMIZER =====================
with tabs[1]:
    st.subheader("🧮 Case Mix Optimizer — Minimize Predicted Billing under Policy Limits")
    st.caption("Uses predicted billing to choose a feasible elective mix that reduces cost exposure.")

    cols = ["_predicted_billing","billing_amount","is_weekend","is_longstay","anomaly_flag",
            "admit_date","insurer","hospital","doctor","condition","length_of_stay","gender"]
    base = df.dropna(subset=["billing_amount"]).copy()
    if "_predicted_billing" not in base.columns:
        base["_predicted_billing"] = base["billing_amount"].astype(float)

    cols_present = [c for c in cols if c in base.columns]
    work_all = base[cols_present].dropna(subset=["_predicted_billing", "billing_amount"])
    if len(work_all) == 0:
        st.error("No records available for optimization. Train the model first or check your data.")
    else:
        max_sample = min(MAX_OPT_SAMPLE, len(work_all))
        sample_n = st.slider("Sampling size (optimization set)", 100, max_sample, min(DEFAULT_OPT_SAMPLE, max_sample), step=50, key="opt_samp")
        work = work_all.sample(n=min(sample_n, len(work_all)), random_state=42).reset_index(drop=True)

        c1, c2, c3, c4 = st.columns(4)
        target_cases = c1.number_input("Target number of cases", min_value=20, max_value=min(1000, len(work)), value=min(120, len(work)))
        max_weekend_pct = c2.slider("Max weekend %", 0, 100, 20, 5)
        max_longstay_pct = c3.slider("Max long-stay %", 0, 100, 12, 2)
        max_anomaly_pct = c4.slider("Max anomaly %", 0, 100, 6, 1)

        c5, c6 = st.columns([1,3])
        use_budget = c5.checkbox("Add budget ceiling", value=False)
        budget_value = c6.number_input(
            "Budget ceiling ($, applies if checked)",
            min_value=0.0,
            value=float(work["_predicted_billing"].sum()*0.6)
        )

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

            pred_total = float(chosen["_predicted_billing"].sum())
            actual_total = float(chosen["billing_amount"].sum())
            naive_topn = work["_predicted_billing"].sort_values().head(int(target_cases)).sum()
            savings_vs_naive = float(naive_topn - pred_total)

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="kpi"><div class="small">Selected Cases</div><h3>{len(chosen):,}</h3></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><div class="small">Predicted Total Billing</div><h3>${pred_total:,.0f}</h3></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><div class="small">Actual Total (selected)</div><h3>${actual_total:,.0f}</h3></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi"><div class="small">Δ vs naive (pred-topN)</div><h3>${savings_vs_naive:,.0f}</h3></div>', unsafe_allow_html=True)

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

            st.download_button("⬇️ Download selected cases (CSV)",
                               data=chosen.to_csv(index=False),
                               file_name="selected_cases.csv",
                               mime="text/csv")

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

# ===================== 3) STAFFING OPTIMIZER =====================
with tabs[2]:
    st.subheader("👩‍⚕️ Staffing Optimizer — RN Targets from Admissions Forecast")
    st.caption("Minimize staffing cost while covering forecast workload with integer shifts (Day/Evening/Night).")

    horizon = st.slider("Forecast horizon (periods)", 7, 28, 14, step=7, key="staff_h")
    fc = admissions_forecast(df, horizon=horizon)
    if fc.empty:
        st.info("Not enough admissions history to forecast.")
    else:
        fig = go.Figure()
        hist = build_timeseries(df, "intake")
        if not hist.empty:
            fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="History", mode="lines"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast", mode="lines"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=50))
        st.plotly_chart(fig, use_container_width=True)

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

            payload = {
                "horizon_periods": horizon,
                "admissions_forecast_avg": float(np.mean(fc["yhat"])),
                "patients_per_RN": float(ratio),
                "total_cost": float(sol["Daily Cost"].sum()),
                "avg_RN_per_period": float(sol["Total_RN"].mean()),
            }
            st.markdown("---")
            ai_write("Staffing Optimizer", payload)

# ===================== 4) TRENDS =====================
with tabs[3]:
    st.subheader("📈 Trends & Explainability")

    if "admit_date" in df.columns and not df["admit_date"].isna().all():
        inferred_freq, _ = infer_ts_granularity(df["admit_date"].dropna().sort_values())
        st.caption(f"Time series frequency inferred: **{inferred_freq}**")
    else:
        inferred_freq = "D"
        st.caption("Time series frequency inferred: **D**")

    c1, c2 = st.columns(2)
    with c1:
        ts_bill = build_timeseries(df, "billing_amount")
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
        "billing_tail": ts_bill.tail(12).to_dict("records") if not ts_bill.empty else None,
        "los_tail": (build_timeseries(df, 'length_of_stay').tail(12).to_dict("records") if 'length_of_stay' in df.columns else None),
        "anomaly_by_group": (mix.to_dict("records") if not mix.empty else None),
    }
    st.markdown("---")
    ai_write("Trends", payload)

st.markdown("---")
st.caption("Built for healthcare operations — simple deployment, big decisions. © 2025")
