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

import os, json, textwrap
from datetime import datetime
from typing import Dict, List

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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize, LpInteger, LpBinary,
    LpStatus, PULP_CBC_CMD
)

# ---------------- CONFIG / THEME ----------------
st.set_page_config(page_title="Hospital Optimization Studio", layout="wide", page_icon="🏥")

RAW_URL = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"

# Calm clinical theme (Sutter/Kaiser friendly blues/teals)
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
</style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Optimization Studio")
st.caption("Price Prediction • Case Mix Optimization • Staffing Targets • Trends • AI Analyst")

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
    for k,v in rename_map.items():
        if k in df.columns: df.rename(columns={k:v}, inplace=True)

    # Types
    if "admit_date" in df: df["admit_date"] = pd.to_datetime(df["admit_date"], errors="coerce")
    if "discharge_date" in df: df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    for c in ["billing_amount","length_of_stay","age"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["admit_date", "billing_amount"]).copy()

    # Feature Engineering
    df["dow"] = df["admit_date"].dt.weekday
    df["month"] = df["admit_date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    if "discharge_date" in df and "admit_date" in df:
        if "length_of_stay" not in df or df["length_of_stay"].isna().all():
            df["length_of_stay"] = (df["discharge_date"] - df["admit_date"]).dt.days
    df["length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")
    df["is_longstay"] = (df["length_of_stay"] > df["length_of_stay"].quantile(0.95)).astype(int)

    for c in ["gender","insurer","hospital","doctor","condition","medication","admission_type"]:
        if c in df: df[c] = df[c].fillna("Unknown")

    # Light anomaly flag
    try:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["anomaly_flag"] = (iso.fit_predict(df[["billing_amount"]]) == -1).astype(int)
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
                if "discharge_date" in df:
                    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
            st.success(f"Loaded {len(df):,} rows from upload.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# ---------------- HELPERS ----------------
def build_timeseries(data: pd.DataFrame, metric: str, freq: str = "D") -> pd.DataFrame:
    if "admit_date" not in data: return pd.DataFrame(columns=["ds","y"])
    idx = data.set_index("admit_date").sort_index()
    if metric == "intake":
        s = idx.assign(_one=1)["_one"].resample(freq).sum().fillna(0.0)
    elif metric == "billing_amount" and "billing_amount" in idx:
        s = idx["billing_amount"].resample(freq).sum().fillna(0.0)
    elif metric == "length_of_stay" and "length_of_stay" in idx:
        s = idx["length_of_stay"].resample(freq).mean().fillna(method="ffill").fillna(0.0)
    else:
        return pd.DataFrame(columns=["ds","y"])
    return pd.DataFrame({"ds": s.index, "y": s.values})

def hw_forecast(s: pd.Series, horizon: int) -> np.ndarray:
    try:
        m = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=7).fit()
    except Exception:
        m = ExponentialSmoothing(s, trend="add").fit()
    return m.forecast(horizon)

def sarimax_forecast(s: pd.Series, horizon: int) -> np.ndarray:
    model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return model.forecast(steps=horizon)

def admissions_forecast(data: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    ts = build_timeseries(data, "intake", "D")
    if ts.empty:
        return pd.DataFrame(columns=["ds","yhat"])
    s = ts.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
    try:
        fc = hw_forecast(s, horizon)
    except Exception:
        fc = sarimax_forecast(s, horizon)
    ds = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"ds": ds, "yhat": np.asarray(fc)})

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

def ai_write(section_title: str, payload: dict):
    client = get_openai_client()
    col1, col2 = st.columns([1,2])
    use_ai = col1.checkbox(f"Use AI for {section_title}", value=(client is not None), key=f"ai_{section_title}")
    analyst = col2.toggle("Analyst mode (more detail)", value=False, key=f"ai_mode_{section_title}")

    if use_ai and client:
        prompt = textwrap.dedent(f"""
        You are a healthcare operations product analyst.
        Write a concise {"executive" if not analyst else "analyst-grade"} narrative (140–220 words)
        for "{section_title}" using ONLY this JSON:
        {json.dumps(payload, default=str)[:6000]}
        Include: What/So-what/Now-what, a tiny "Performance & Use Case" table,
        and 3–5 concrete recommendations with owners and suggested SLAs.
        """).strip()
        try:
            rsp = client.chat.completions.create(
                model=(st.secrets.get("PREFERRED_OPENAI_MODEL") or "gpt-4o-mini"),
                messages=[{"role": "system", "content": "Be precise, concise, and actionable."},
                          {"role": "user", "content": prompt}],
                temperature=0.2
            )
            st.markdown(rsp.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
            st.json(payload)
    else:
        st.markdown(f"**Deterministic summary — {section_title}**")
        st.json(payload)
        st.caption("Add OPENAI_API_KEY in Secrets/env to enable AI narratives.")

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
    use_cols = num_cols + cat_cols + ["billing_amount"]
    data = df.dropna(subset=["billing_amount"])[use_cols].copy()

    if len(data) < 100:
        st.info("Not enough rows to train a robust model.")
    else:
        X = data.drop(columns=["billing_amount"])
        y = data["billing_amount"].astype(float)

        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
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

        # Permutation importance (fast & robust)
        st.markdown("#### Top Drivers (Permutation Importance)")
        try:
            imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            # Get feature names from ColumnTransformer
            ohe = model.named_steps["pre"].named_transformers_["cat"]
            cat_names = []
            if hasattr(ohe, "get_feature_names_out"):
                cat_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names = num_cols + cat_names
            importances = pd.DataFrame({
                "feature": feature_names[:len(imp.importances_mean)],
                "importance": imp.importances_mean[:len(feature_names)]
            }).sort_values("importance", ascending=False).head(15)
            fig_imp = px.bar(importances, x="feature", y="importance", title="Top 15 Features", height=380)
            fig_imp.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info("Permutation importance unavailable; continuing.")

        # Save predictions back to df for optimization
        try:
            df["_predicted_billing"] = model.predict(df[X.columns].fillna(0))
        except Exception:
            # If some columns missing after upload, align
            aligned = pd.DataFrame(columns=X.columns)
            aligned = pd.concat([aligned, df.reindex(columns=X.columns)], axis=0).iloc[len(df)*-1:]
            df["_predicted_billing"] = model.predict(aligned.fillna(0))

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
    sample_n = st.slider("Sampling size (optimization set)", 100, 1000, 250, step=50, key="opt_samp")
    cols = ["_predicted_billing","billing_amount","is_weekend","is_longstay","anomaly_flag",
            "admit_date","insurer","hospital","doctor","condition","length_of_stay","gender"]
    base = df.dropna(subset=["billing_amount"]).copy()
    if "_predicted_billing" not in base.columns:
        # Fallback: if prediction not computed (e.g., tiny dataset), use actual billing
        base["_predicted_billing"] = base["billing_amount"].astype(float)
    work = base[cols].dropna().sample(n=min(sample_n, len(base)), random_state=42).reset_index(drop=True)

    # Policy controls
    c1, c2, c3, c4 = st.columns(4)
    target_cases = c1.number_input("Target number of cases", min_value=20, max_value=min(800, len(work)), value=min(120, len(work)))
    max_weekend_pct = c2.slider("Max weekend %", 0, 100, 20, 5)
    max_longstay_pct = c3.slider("Max long-stay %", 0, 100, 12, 2)
    max_anomaly_pct = c4.slider("Max anomaly %", 0, 100, 6, 1)

    # Optional budget ceiling (soft; can be toggled on)
    c5, c6 = st.columns([1,3])
    use_budget = c5.checkbox("Add budget ceiling", value=False)
    budget_value = c6.number_input("Budget ceiling ($, applies if checked)", min_value=0.0, value=float(work["_predicted_billing"].sum()*0.6))

    # MILP: minimize predicted total cost subject to constraints
    model = LpProblem("CaseMix_MinPredictedCost", LpMinimize)
    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in range(len(work))]
    cost = work["_predicted_billing"].values

    model += lpSum(cost[i] * x[i] for i in range(len(work)))
    model += lpSum(x) == int(target_cases)
    model += lpSum(work["is_weekend"].values[i] * x[i] for i in range(len(work))) <= (max_weekend_pct/100.0) * target_cases
    model += lpSum(work["is_longstay"].values[i] * x[i] for i in range(len(work))) <= (max_longstay_pct/100.0) * target_cases
    model += lpSum(work["anomaly_flag"].values[i] * x[i] for i in range(len(work))) <= (max_anomaly_pct/100.0) * target_cases
    if use_budget:
        model += lpSum(cost[i] * x[i] for i in range(len(work))) <= budget_value

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
        baseline_equal_count = work["_predicted_billing"].sort_values().head(int(target_cases)).sum()
        savings_vs_naive = baseline_equal_count - pred_total  # how much better than naive "cheapest by prediction" top-N in sample

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi"><div class="small">Selected Cases</div><h3>{len(chosen):,}</h3></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="small">Predicted Total Billing</div><h3>${pred_total:,.0f}</h3></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="small">Actual Total (selected)</div><h3>${actual_total:,.0f}</h3></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi"><div class="small">Δ vs naive (pred-topN)</div><h3>${savings_vs_naive:,.0f}</h3></div>', unsafe_allow_html=True)

        # Composition charts
        cA, cB = st.columns(2)
        with cA:
            by_insurer = chosen.groupby("insurer")["_predicted_billing"].sum().sort_values(ascending=False).head(12).reset_index()
            fig = px.bar(by_insurer, x="insurer", y="_predicted_billing", title="Predicted Cost by Insurer")
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            by_cond = chosen.groupby("condition")["_predicted_billing"].sum().sort_values(ascending=False).head(12).reset_index()
            fig = px.bar(by_cond, x="condition", y="_predicted_billing", title="Predicted Cost by Condition")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Selected Cases (sample)")
        show_cols = ["admit_date","hospital","doctor","insurer","condition","_predicted_billing","billing_amount",
                     "length_of_stay","is_weekend","is_longstay","anomaly_flag"]
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
            "composition": {"by_insurer": by_insurer.to_dict("records"), "by_condition": by_cond.to_dict("records")},
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

    # Forecast horizon
    horizon = st.slider("Forecast horizon (days)", 7, 28, 14, step=7, key="staff_h")
    fc = admissions_forecast(df, horizon=horizon)
    if fc.empty:
        st.info("Not enough admissions history to forecast.")
    else:
        # Display forecast
        fig = go.Figure()
        hist = build_timeseries(df, "intake", "D")
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
            k2.markdown(f'<div class="kpi"><div class="small">Total Cost ({horizon}d)</div><h3>${sol["Daily Cost"].sum():,.0f}</h3></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><div class="small">Patients per RN</div><h3>{ratio:.1f}</h3></div>', unsafe_allow_html=True)

            st.markdown("#### RN Targets by Day")
            st.dataframe(sol, use_container_width=True, hide_index=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=sol["Date"], y=sol["RN_Day"], name="Day"))
            fig2.add_trace(go.Bar(x=sol["Date"], y=sol["RN_Evening"], name="Evening"))
            fig2.add_trace(go.Bar(x=sol["Date"], y=sol["RN_Night"], name="Night"))
            fig2.update_layout(barmode="stack", height=380, title="RN Staffing Plan (stacked by shift)")
            st.plotly_chart(fig2, use_container_width=True)

            # AI
            payload = {
                "horizon_days": horizon,
                "admissions_forecast_avg": float(np.mean(fc["yhat"])),
                "patients_per_RN": float(ratio),
                "total_cost": float(sol["Daily Cost"].sum()),
                "avg_RN_per_day": float(sol["Total_RN"].mean()),
            }
            st.markdown("---")
            ai_write("Staffing Optimizer", payload)

# ======================================================================================
# 4) TRENDS
# ======================================================================================
with tabs[3]:
    st.subheader("📈 Trends & Explainability")
    c1, c2 = st.columns(2)
    with c1:
        ts_bill = build_timeseries(df, "billing_amount", "W")
        if ts_bill.empty:
            st.info("No billing time series available.")
        else:
            fig = px.line(ts_bill, x="ds", y="y", title="Billing — Weekly")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "length_of_stay" in df:
            ts_los = build_timeseries(df, "length_of_stay", "W")
            fig = px.line(ts_los, x="ds", y="y", title="Avg LOS — Weekly")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LOS field available.")

    st.markdown("#### Current Anomaly Mix")
    mix = df.groupby("insurer").agg(
        total_cost=("billing_amount","sum"),
        n=("billing_amount","count"),
        anomaly_rate=("anomaly_flag","mean")
    ).reset_index().sort_values("total_cost", ascending=False).head(15)
    fig = px.bar(mix, x="insurer", y="total_cost", color="anomaly_rate",
                 title="Top Insurers by Cost (color = anomaly rate)", labels={"total_cost":"Total Billing"})
    st.plotly_chart(fig, use_container_width=True)

    payload = {
        "billing_weekly_tail": ts_bill.tail(12).to_dict("records") if not ts_bill.empty else None,
        "los_weekly_tail": ts_los.tail(12).to_dict("records") if "length_of_stay" in df else None,
        "anomaly_by_insurer": mix.to_dict("records"),
    }
    st.markdown("---")
    ai_write("Trends", payload)

# ======================= FOOTER =======================
st.markdown("---")
st.caption("Built for healthcare operations — simple deployment, big decisions. © 2025")
