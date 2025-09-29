# app.py
# Hospital Optimization Suite — v7
# - Per-tab filters, no sidebar
# - Executive-friendly problem writeups (implications + models + how they help)
# - LP terminology explained in plain English on each tab
# - Consistent ▶️ Run Optimization buttons
# - AI summaries via OpenAI (only GPT-4.0 and GPT-3.5) with Local fallback
# - Summaries recompute from FILTERED data + latest solver outputs
# - Editable “Run Custom Code” on each tab (guarded)
# - Robust staffing model (no IndexError) and CSV auto-load from /mnt/data

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Optimization (LP)
from pulp import (
    LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
)

# ---- OpenAI client (STRICT: only gpt-4.0 & gpt-3.5-turbo) -------------------
OPENAI_READY = False
OPENAI_MODELS = ["Local (no-LLM)", "gpt-4.0", "gpt-3.5-turbo"]
try:
    from openai import OpenAI
    _openai_key = os.environ.get("OPENAI_API_KEY", None)
    if not _openai_key and hasattr(st, "secrets"):
        _openai_key = st.secrets.get("OPENAI_API_KEY", None)
    if _openai_key:
        client = OpenAI(api_key=_openai_key)
        OPENAI_READY = True
except Exception:
    OPENAI_READY = False

# -----------------------------
# Page config & theming
# -----------------------------
st.set_page_config(
    page_title="Hospital Optimization Suite",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
  .main-header { font-size: 2.25rem; color: #0f172a; font-weight: 800; margin: 0 0 .25rem 0;}
  .subheader { color: #334155; margin-bottom: 1.0rem; font-size: .95rem;}
  .section-header { color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin: 18px 0 10px 0;}
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 1px 1px rgba(0,0,0,.03);}
  .kpi { background-color: #f8fafc; padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0; }
  .muted { color: #64748b; font-size: .9rem;}
  .results { background-color: #f8fafc; padding: 12px; border-radius: 12px; border: 2px solid #22c55e; }
  .ok { background-color: #ecfdf5; padding: 12px; border-left: 5px solid #10b981; border-radius: 10px;}
  .warn { background-color: #fff7ed; padding: 12px; border-left: 5px solid #fb923c; border-radius: 10px;}
  .info { background-color: #eff6ff; padding: 12px; border-left: 5px solid #3b82f6; border-radius: 10px;}
  .small { font-size: .85rem; color: #475569;}
  .danger { color:#b91c1c; }
  .codeblock { background: #0b1021; color: #e5e7eb; padding: 12px; border-radius: 8px; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def normalize_view(v: str) -> str:
    if not v: return "executive"
    v = v.lower().strip().replace("-", " ").replace("_", " ")
    if "exec" in v: return "executive"
    if "tech" in v or "deep" in v or "analyst" in v: return "technical"
    return "executive"

@st.cache_data(show_spinner=False)
def try_load_default_csv():
    # Prefer user upload; else try reference CSV; else None (then synth)
    path = "/mnt/data/modified_healthcare_dataset.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def generate_synthetic():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    departments = ['ICU', 'Emergency', 'General', 'Pediatrics', 'Surgery']

    # Bed data
    rows = []
    cap_map = {'ICU': 20, 'Emergency': 30, 'General': 100, 'Pediatrics': 25, 'Surgery': 40}
    for d in dates:
        for dept in departments:
            cap = cap_map[dept]
            occ = np.clip(rng.normal(0.8, 0.15) * cap, 0, cap)
            rows.append({"date": d, "department": dept, "capacity": cap,
                         "occupied": int(occ), "available": cap - int(occ),
                         "utilization_rate": occ/cap,
                         "admission_type":"Scheduled" if rng.random()>.4 else "Emergency",
                         "condition": rng.choice(["Cardiac","Respiratory","Surgical","Pediatric"]),
                         "hospital": rng.choice(["North","Central","South"])})
    bed_df = pd.DataFrame(rows)

    # Staff (last 30 days)
    staff_roles = ['Nurses', 'Doctors', 'Technicians', 'Support Staff']
    shifts = ['Morning', 'Afternoon', 'Night']
    staff_rows = []
    reqs = {'Nurses': 15, 'Doctors': 8, 'Technicians': 5, 'Support Staff': 10}
    for d in dates[-30:]:
        for role in staff_roles:
            for shift in shifts:
                req = reqs[role]
                avail = rng.poisson(req * 0.9)
                staff_rows.append({"date": d, "role": role, "shift": shift,
                                   "required": req, "available": int(avail),
                                   "shortage": max(0, req - int(avail)),
                                   "hospital": rng.choice(["North","Central","South"]),
                                   "department": rng.choice(departments)})
    staff_df = pd.DataFrame(staff_rows)

    # Resources
    resources = ['Ventilators', 'X-Ray Machines', 'CT Scanners', 'Wheelchairs', 'Monitors']
    totals = {'Ventilators': 25, 'X-Ray Machines': 5, 'CT Scanners': 3, 'Wheelchairs': 50, 'Monitors': 80}
    res_rows = []
    for r in resources:
        total = totals[r]
        in_use = int(rng.integers(int(total*0.4), int(total*0.9)))
        maint = int(rng.integers(0, 3))
        res_rows.append({"resource": r, "total": total, "in_use": in_use,
                         "maintenance": maint, "available": total-in_use-maint,
                         "utilization_rate": in_use/total, "hospital": rng.choice(["North","Central","South"])})
    resource_df = pd.DataFrame(res_rows)
    return bed_df, staff_df, resource_df

def split_or_infer(df: pd.DataFrame):
    df = df.rename(columns={c: c.lower() for c in df.columns})
    optional = ["hospital","admission_type","condition","department"]

    bed_cols = {"date","department","capacity","occupied","available","utilization_rate"}
    staff_cols = {"date","role","shift","required","available","shortage"}
    res_cols = {"resource","total","in_use","maintenance","available","utilization_rate"}

    cols = set(df.columns)
    bed_df = df[list(bed_cols.union(set(c for c in optional if c in df.columns)))] if bed_cols.issubset(cols) else None
    staff_df = df[list(staff_cols.union(set(c for c in optional if c in df.columns)))] if staff_cols.issubset(cols) else None
    resource_df = df[list(res_cols.union(set(c for c in optional if c in df.columns)))] if res_cols.issubset(cols) else None

    if bed_df is None or staff_df is None or resource_df is None:
        sb, ss, sr = generate_synthetic()
        bed_df = bed_df if bed_df is not None else sb
        staff_df = staff_df if staff_df is not None else ss
        resource_df = resource_df if resource_df is not None else sr

    if "date" in bed_df.columns: bed_df["date"] = pd.to_datetime(bed_df["date"], errors="coerce")
    if "date" in staff_df.columns: staff_df["date"] = pd.to_datetime(staff_df["date"], errors="coerce")
    return bed_df, staff_df, resource_df

def apply_filters(bed_df, staff_df, resource_df, hospital=None, department=None, admission=None, condition=None):
    """Apply consistent filters to tables if columns exist; return filtered copies."""
    def ftab(tab, col, vals):
        if col in tab.columns and vals:
            return tab[tab[col].isin(vals)]
        return tab

    b, s, r = bed_df.copy(), staff_df.copy(), resource_df.copy()
    b = ftab(b, "hospital", hospital); s = ftab(s, "hospital", hospital); r = ftab(r, "hospital", hospital)
    b = ftab(b, "department", department); s = ftab(s, "department", department) if "department" in s.columns else s
    b = ftab(b, "admission_type", admission)
    b = ftab(b, "condition", condition)
    return b, s, r

def filter_widgets(label_prefix, bed_df, staff_df, resource_df):
    """Render multiselect filters valid for this tab; return filtered tables."""
    def opts(tab, col):
        return sorted(tab[col].dropna().unique()) if col in tab.columns else []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hospitals = st.multiselect(f"{label_prefix} — Hospital/Facility", opts(bed_df,"hospital") or opts(staff_df,"hospital") or opts(resource_df,"hospital"))
    with c2:
        departments = st.multiselect(f"{label_prefix} — Department", opts(bed_df,"department") or opts(staff_df,"department"))
    with c3:
        admissions = st.multiselect(f"{label_prefix} — Admission Type", opts(bed_df,"admission_type"))
    with c4:
        conditions = st.multiselect(f"{label_prefix} — Condition", opts(bed_df,"condition"))

    return apply_filters(bed_df, staff_df, resource_df,
                         hospital=hospitals, department=departments,
                         admission=admissions, condition=conditions)

# -----------------------------
# Optimization models (robust)
# -----------------------------
def bed_allocation_basic(bed_df: pd.DataFrame):
    depts = bed_df['department'].dropna().unique()
    if len(depts) == 0:
        return {"status":"Infeasible","objective_value":None,"allocation":{}}
    cap_by = bed_df.groupby('department', dropna=True)['capacity'].first().astype(float).to_dict()
    prob = LpProblem("Bed_Allocation_Basic", LpMaximize)
    x = {d: LpVariable(f"beds_{d}", lowBound=0, upBound=cap_by[d]) for d in depts}
    prob += lpSum([x[d] for d in depts])
    for d in depts: prob += x[d] <= cap_by[d]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
            "allocation": {d: float(value(x[d])) for d in depts}}

def bed_allocation_demand_based(bed_df: pd.DataFrame, demand_multipliers=None, weights=None):
    depts = bed_df['department'].dropna().unique()
    if len(depts) == 0:
        return {"status":"Infeasible","objective_value":None,"allocation":{},"shortages":{}}
    cap = bed_df.groupby('department', dropna=True)['capacity'].first().astype(float).to_dict()
    if demand_multipliers is None:
        util = bed_df.groupby('department')['utilization_rate'].mean().to_dict()
        demand_multipliers = {d: float(np.clip(util.get(d, 0.8)*1.05 + 0.05, 0.85, 1.35)) for d in depts}
    if weights is None:
        weights = {d: (10 if d in ("ICU","Emergency") else 6) for d in depts}
    prob = LpProblem("Bed_Allocation_Demand", LpMinimize)
    x = {d: LpVariable(f"beds_{d}", lowBound=0, upBound=cap[d]) for d in depts}
    s = {d: LpVariable(f"short_{d}", lowBound=0) for d in depts}
    prob += lpSum([weights[d]*s[d] for d in depts])
    for d in depts:
        expected = cap[d] * demand_multipliers[d]
        prob += x[d] <= cap[d]
        prob += s[d] >= expected - x[d]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
            "allocation": {d: float(value(x[d])) for d in depts},
            "shortages": {d: float(value(s[d])) for d in depts},
            "demand_multipliers": demand_multipliers, "weights": weights}

def staff_scheduling_basic(staff_df: pd.DataFrame, overtime_costs=None):
    roles = sorted(staff_df['role'].dropna().unique())
    shifts = sorted(staff_df['shift'].dropna().unique())
    if len(roles) == 0 or len(shifts) == 0:
        return {"status":"Infeasible","total_cost":None,"assignments":{},"overtime":{},"overtime_costs":overtime_costs or {}}
    agg = staff_df.groupby(['role','shift']).agg(required=('required','mean'), available=('available','mean'))
    idx = pd.MultiIndex.from_product([roles, shifts], names=['role','shift'])
    agg_full = agg.reindex(idx).fillna({'required':0.0,'available':0.0})
    req = {(r,s): float(v) for (r,s), v in agg_full['required'].items()}
    ava = {(r,s): float(v) for (r,s), v in agg_full['available'].items()}
    if overtime_costs is None:
        overtime_costs = {'Nurses':45,'Doctors':80,'Technicians':35,'Support Staff':25}
    prob = LpProblem("Staff_Scheduling", LpMinimize)
    assign = {(r,s): LpVariable(f"assign_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    ot = {(r,s): LpVariable(f"ot_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    prob += lpSum([overtime_costs.get(r,40)*ot[(r,s)] for r in roles for s in shifts])
    for r in roles:
        for s in shifts:
            prob += assign[(r,s)] + ot[(r,s)] >= req[(r,s)]
            prob += assign[(r,s)] <= ava[(r,s)]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "total_cost": float(value(prob.objective)) if prob.status == 1 else None,
            "assignments": {f"{r}_{s}": float(value(assign[(r,s)])) for r in roles for s in shifts},
            "overtime": {f"{r}_{s}": float(value(ot[(r,s)])) for r in roles for s in shifts},
            "overtime_costs": overtime_costs}

def resource_optimization_basic(resource_df: pd.DataFrame):
    resources = resource_df['resource'].dropna().unique()
    if len(resources) == 0:
        return {"status":"Infeasible","objective_value":None,"allocation":{}}
    total = resource_df.set_index('resource')['total'].astype(float).to_dict()
    maint = resource_df.set_index('resource')['maintenance'].astype(float).to_dict()
    prob = LpProblem("Resource_Optimization", LpMaximize)
    x = {r: LpVariable(f"alloc_{r}", lowBound=0, upBound=total[r]) for r in resources}
    prob += lpSum([x[r] for r in resources])
    for r in resources:
        prob += x[r] <= total[r] - maint[r]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
            "allocation": {r: float(value(x[r])) for r in resources}}

# -----------------------------
# AI summaries — OpenAI (with strict model list) + fallback
# -----------------------------
def ai_summary_with_openai(model_name: str, role: str, module: str, payload: dict) -> str:
    if model_name not in ["gpt-4.0", "gpt-3.5-turbo"]:
        raise ValueError("Model not allowed. Choose gpt-4.0 or gpt-3.5-turbo.")
    system = (
        "You are a concise hospital operations analyst. "
        "Write in the selected role voice (Executive or Technical). "
        "Use only the data provided; do not invent numbers."
    )
    user = {"role": role, "module": module, "data": payload}
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)}
        ],
        temperature=0.2,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()

def generate_ai_summary_local(module, results, view_type, bed_df=None, staff_df=None, resource_df=None):
    """Deterministic fallback (no LLM)."""
    vt = normalize_view(view_type)
    def fmt_money(x): return f"${x:,.0f}"
    def pct(x): return f"{x*100:.1f}%"

    if module == "bed_allocation":
        alloc = results.get("allocation", {}) or {}
        total_alloc = sum(alloc.values())
        total_cap = bed_df.groupby('department')['capacity'].first().sum() if (bed_df is not None and len(bed_df)) else 0
        alloc_util = (total_alloc/total_cap) if total_cap>0 else None
        shortages = results.get("shortages", {}) or {}
        worst_short = sorted(shortages.items(), key=lambda x: x[1], reverse=True)[:2] if shortages else []
        top_busy = []
        if bed_df is not None and len(bed_df):
            mu = bed_df.groupby('department')['utilization_rate'].mean().sort_values(ascending=False)
            top_busy = list(mu.index[:2])
            baseline = float(mu.mean())
        else:
            baseline = 0.0
        eff_gain = max(0.0, (alloc_util or baseline) - baseline)
        est = eff_gain * (total_cap or 0) * 500

        if vt == "executive":
            bullets = []
            if alloc_util is not None:
                bullets.append(f"Projected post-optimization bed utilization: {pct(alloc_util)}.")
            if worst_short:
                bullets.append("Residual pressure: " + ", ".join([f"{d} ({s:.1f} beds short)" for d,s in worst_short]) + ".")
            if top_busy:
                bullets.append("Sustained demand in: " + ", ".join(top_busy) + ".")
            if est > 0:
                bullets.append(f"Estimated annual upside (throughput/diversion): {fmt_money(est)}.")
            recs = ["Daily load-balancing targets", "Dynamic bed board (ICU/ED surge)", "LOS-based discharge readiness"]
            return "\n".join([f"- {b}" for b in bullets] + ["**Recommendations:**"] + [f"- {r}" for r in recs])
        else:
            details = {
                "Solver Status": results.get("status"),
                "Objective": results.get("objective_value"),
                "Demand Multipliers": results.get("demand_multipliers"),
                "Weights": results.get("weights"),
                "Top Shortages": worst_short,
                "Baseline Utilization (filtered mean)": baseline
            }
            return str(details)

    if module == "staff_scheduling":
        ov = results.get("overtime", {}) or {}
        over_items = sorted(ov.items(), key=lambda kv: kv[1], reverse=True)[:3]
        fill_rate = None
        if staff_df is not None and len(staff_df) and staff_df['required'].sum()>0:
            fill_rate = staff_df['available'].sum()/staff_df['required'].sum()
        if vt == "executive":
            bullets = []
            if fill_rate is not None: bullets.append(f"Baseline fill rate: {fill_rate*100:.1f}%.")
            if over_items: bullets.append("Overtime hotspots: " + ", ".join([f"{k.replace('_',' ')} ({v:.1f})" for k,v in over_items]))
            tot = results.get("total_cost")
            if tot is not None: bullets.append(f"Optimized weekly OT cost: {fmt_money(tot)}.")
            recs = ["Night/ED float pool", "Preference bidding", "Rebalance ICU night skill mix"]
            return "\n".join([f"- {b}" for b in bullets] + ["**Recommendations:**"] + [f"- {r}" for r in recs])
        else:
            details = {
                "Solver Status": results.get("status"),
                "Weekly OT Cost": results.get("total_cost"),
                "Top OT Cells": over_items,
                "Cost Weights": results.get("overtime_costs")
            }
            return str(details)

    if module == "resource_optimization":
        post = None; top = []
        if resource_df is not None and results.get("allocation"):
            alloc = pd.Series(results["allocation"])
            total = resource_df.set_index('resource')['total']
            post = (alloc/total).sort_values(ascending=False)
            top = list(post.index[:3])
        if vt == "executive":
            bullets = []
            if post is not None and len(post):
                bullets.append("Post-optimization device utilization leaders: " + ", ".join(top))
                bullets.append(f"Average device utilization after allocation: {float(post.mean())*100:.1f}%")
            recs = ["Swap queue for low-velocity devices", "Maintenance outside peaks", "RTLS beacons on high-value assets"]
            return "\n".join([f"- {b}" for b in bullets] + ["**Recommendations:**"] + [f"- {r}" for r in recs])
        else:
            details = {
                "Solver Status": results.get("status"),
                "Objective": results.get("objective_value"),
                "Allocation": results.get("allocation")
            }
            return str(details)

    return "No details."

def generate_ai_summary(module, results, view_type, llm_model, bed_df=None, staff_df=None, resource_df=None):
    """
    High-level wrapper:
      - Build a compact 'payload' with the exact data for the LLM
      - If model is 'Local' or no OpenAI key, use deterministic local summary
      - Otherwise call OpenAI and fall back to local on error
    """
    role = "Executive" if normalize_view(view_type) == "executive" else "Technical"
    payload = {"role": role, "module": module, "results": results}

    if module == "bed_allocation" and bed_df is not None:
        payload["kpis"] = {
            "departments": int(bed_df['department'].nunique()),
            "avg_util": float(bed_df['utilization_rate'].mean()) if len(bed_df) else 0.0,
            "total_capacity": float(bed_df.groupby('department')['capacity'].first().sum()) if len(bed_df) else 0.0
        }
    if module == "staff_scheduling" and staff_df is not None:
        payload["kpis"] = {
            "total_required": float(staff_df['required'].sum()) if 'required' in staff_df.columns else 0.0,
            "total_available": float(staff_df['available'].sum()) if 'available' in staff_df.columns else 0.0
        }
    if module == "resource_optimization" and resource_df is not None:
        payload["kpis"] = {"avg_util": float(resource_df['utilization_rate'].mean()) if len(resource_df) else 0.0}

    # Local path?
    if (llm_model not in OPENAI_MODELS) or (llm_model == "Local (no-LLM)") or (not OPENAI_READY):
        return generate_ai_summary_local(module, results, view_type, bed_df, staff_df, resource_df)

    # OpenAI path with fallback
    try:
        return ai_summary_with_openai(llm_model, role, module, payload)
    except Exception as e:
        st.warning(f"AI summary fell back to local ({e}).")
        return generate_ai_summary_local(module, results, view_type, bed_df, staff_df, resource_df)

# -----------------------------
# Data load
# -----------------------------
default_df = try_load_default_csv()
uploaded = st.file_uploader("📥 Upload CSV (optional) — else we'll try /mnt/data/modified_healthcare_dataset.csv, then a demo dataset.", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded); st.success("File uploaded.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}"); raw_df = None
elif default_df is not None:
    raw_df = default_df
else:
    raw_df = None

if raw_df is None:
    bed_df, staff_df, resource_df = generate_synthetic()
else:
    bed_df, staff_df, resource_df = split_or_infer(raw_df)

# -----------------------------
# Header + tabs
# -----------------------------
st.markdown('<div class="main-header">🏥 Hospital Optimization Suite</div>', unsafe_allow_html=True)
sub_left, sub_right = st.columns([0.7, 0.3])
with sub_left:
    st.markdown('<div class="subheader">Executives see outcomes. Analysts see levers. Everyone sees the same data.</div>', unsafe_allow_html=True)
with sub_right:
    # Global AI model picker (strict list)
    default_model = "gpt-4.0" if OPENAI_READY else "Local (no-LLM)"
    llm_model = st.selectbox("AI summary model", OPENAI_MODELS, index=OPENAI_MODELS.index(default_model))
    if not OPENAI_READY and llm_model != "Local (no-LLM)":
        st.info("No OpenAI API key found — using local summaries.", icon="ℹ️")

tabs = st.tabs(["🏠 Overview", "🛏️ Beds", "👥 Staff", "🧰 Resources", "🧭 Notes & Export"])

# -----------------------------
# Tab 0 — Overview
# -----------------------------
with tabs[0]:
    st.markdown('<div class="section-header">🔎 Global Filters (Overview)</div>', unsafe_allow_html=True)
    bF, sF, rF = filter_widgets("Overview", bed_df, staff_df, resource_df)

    st.markdown('<div class="section-header">📈 KPIs (Filtered)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_u = bF['utilization_rate'].mean() if len(bF) else 0.0
        st.markdown(f'<div class="kpi"><div class="muted">Bed Utilization</div><h3>{avg_u:.1%}</h3><div class="muted">Avg across filtered depts</div></div>', unsafe_allow_html=True)
    with c2:
        staff_short = int(sF['shortage'].sum()) if len(sF) else 0
        st.markdown(f'<div class="kpi"><div class="muted">Staff Shortage</div><h3>{staff_short}</h3><div class="muted">Shifts understaffed (30d)</div></div>', unsafe_allow_html=True)
    with c3:
        avg_res = rF['utilization_rate'].mean() if len(rF) else 0.0
        st.markdown(f'<div class="kpi"><div class="muted">Resource Utilization</div><h3>{avg_res:.1%}</h3><div class="muted">Avg devices in use</div></div>', unsafe_allow_html=True)
    with c4:
        tot_cap = bF.groupby('department')['capacity'].first().sum() if len(bF) else 0
        potential = max(0, (avg_u-0.7)) * tot_cap * 600 if tot_cap else 0
        st.markdown(f'<div class="kpi"><div class="muted">Potential Upside</div><h3>${potential:,.0f}</h3><div class="muted">Throughput & overtime</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Trends</div>', unsafe_allow_html=True)
    A, B = st.columns(2)
    with A:
        if len(bF):
            daily = bF.groupby(['date','department'])['utilization_rate'].mean().reset_index()
            fig = px.line(daily, x='date', y='utilization_rate', color='department', title="Bed Utilization by Department")
            fig.update_layout(yaxis_tickformat='.0%', height=360, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bed data in current filter.")
    with B:
        if len(rF):
            res_fig = go.Figure()
            res_fig.add_trace(go.Bar(name='In Use', x=rF['resource'], y=rF['in_use']))
            res_fig.add_trace(go.Bar(name='Available', x=rF['resource'], y=rF['available']))
            res_fig.add_trace(go.Bar(name='Maintenance', x=rF['resource'], y=rF['maintenance']))
            res_fig.update_layout(barmode='stack', title="Resource Status", height=360, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(res_fig, use_container_width=True)
        else:
            st.info("No resource data in current filter.")

# -----------------------------
# Tab 1 — Beds
# -----------------------------
with tabs[1]:
    st.markdown('<div class="section-header">🛏️ Bed Allocation — Plain-English Problem & LP Models</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<b>Problem in one sentence:</b> Beds aren’t always where the demand is, so patients wait and EDs divert.<br><br>
<b>Business impact if we do nothing:</b> longer throughput time, diversion penalties, lost revenue, poorer experience.<br><br>
<b>What we’re optimizing (LP in simple words):</b> We choose how many beds each department should operate. We either maximize beds actively used, or we minimize the expected shortfall where demand exceeds available beds.<br><br>
<b>Models we use & how they help:</b>
<ul>
  <li><b>Basic Utilization (maximize):</b> pushes each unit toward its capacity. Good for a quick “fill what you can” plan.</li>
  <li><b>Demand-Based (minimize weighted shortage):</b> estimates demand per unit and minimizes shortfalls; we can give ICU/ED extra weight to protect critical care.</li>
</ul>
<b>Constraints (the rules):</b> You can’t allocate more beds than a unit’s capacity; allocations can’t be negative.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔎 Filters (Beds)</div>', unsafe_allow_html=True)
    bBed, sBed, rBed = filter_widgets("Beds", bed_df, staff_df, resource_df)

    st.markdown("#### Configure & Run")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model_b = st.radio("Optimization approach", ["Basic Utilization", "Demand-Based (weighted shortages)"], key="bed_model")
    with c2:
        ai_view_b = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], key="ai_b")
    with c3:
        demand_stress = st.slider("Demand stress (+/- %)", -20, 40, 0)
    with c4:
        icu_bump = st.slider("ICU priority bump", 0, 5, 2)

    if st.button("▶️ Run Optimization", key="run_beds"):
        with st.spinner("Solving bed allocation on current filters…"):
            if model_b.startswith("Basic"):
                bed_res = bed_allocation_basic(bBed)
            else:
                util = bBed.groupby('department')['utilization_rate'].mean() if len(bBed) else pd.Series(dtype=float)
                dm = {d: float(np.clip((u if not np.isnan(u) else 0.8) * (1 + demand_stress/100.0), 0.8, 1.4)) for d,u in util.items()}
                weights = {d: (10 + icu_bump if d=="ICU" else (9 if d=="Emergency" else 6)) for d in util.index}
                bed_res = bed_allocation_demand_based(bBed, dm if dm else None, weights if weights else None)
            st.session_state["bed_results"] = bed_res
            st.session_state["bed_filtered"] = bBed.copy()

    if "bed_results" in st.session_state:
        res = st.session_state["bed_results"]
        b_used = st.session_state.get("bed_filtered", bBed)
        st.markdown('<div class="results">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Solver", res.get("status","?"))
        with c2: st.metric("Objective", f"{res.get('objective_value'):.2f}" if res.get('objective_value') is not None else "—")
        with c3: st.metric("Departments", b_used['department'].nunique() if len(b_used) else 0)
        with c4: st.metric("Total Capacity", int(b_used.groupby('department')['capacity'].first().sum()) if len(b_used) else 0)
        st.markdown('</div>', unsafe_allow_html=True)

        alloc_df = pd.DataFrame([(k,v) for k,v in (res.get("allocation",{}) or {}).items()], columns=["Department","Allocated Beds"])
        if len(alloc_df):
            fig = px.bar(alloc_df, x="Department", y="Allocated Beds", title="Optimal Bed Allocation")
            fig.update_layout(height=380, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation (check filters and capacities).")

        st.markdown("### AI Summary")
        summary_text = generate_ai_summary("bed_allocation", res, ai_view_b, llm_model, bed_df=b_used)
        st.markdown(summary_text)

        # Export
        md = f"# Bed Allocation Report\n\n**Status:** {res.get('status')}\n\n**Objective:** {res.get('objective_value')}\n\n## Allocation\n"
        for d, v in (res.get("allocation",{}) or {}).items():
            md += f"- {d}: {v:.1f}\n"
        if len(b_used):
            md += "\n## Filter Context\n"
            for col in ["hospital","department","admission_type","condition"]:
                if col in b_used.columns:
                    vals = ", ".join(sorted([str(x) for x in b_used[col].dropna().unique()]))
                    md += f"- {col}: {vals}\n"
        st.download_button("⬇️ Download Bed Report (Markdown)", data=md, file_name="bed_allocation_report.md", type="secondary")

    with st.expander("📘 LP in 30 seconds + the math we used"):
        st.markdown("""
**What’s LP?** We pick numbers (decisions) to maximize a benefit or minimize a cost, subject to straight-line rules (constraints).  
**Why it helps here:** it finds the best bed split across units given capacity and demand — no guesswork.

**Demand-Based math:**  
- Variables: xᵢ (beds to unit i), sᵢ (shortage in unit i)  
- Objective: minimize Σ wᵢ·sᵢ  
- Constraints: xᵢ ≤ capacityᵢ, sᵢ ≥ demandᵢ − xᵢ, xᵢ, sᵢ ≥ 0
""")

    with st.expander("🧑‍💻 Advanced — Run Custom Bed Model (unsafe: executes your code)"):
        st.markdown('<span class="small danger">This executes on the server. Use only if you understand the risks.</span>', unsafe_allow_html=True)
        enable = st.checkbox("I understand the risks and want to run custom code", key="bed_custom_enable")
        default_bed_code = """# Define custom_bed_model(df) -> dict like the built-ins.
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
import numpy as np, pandas as pd
def custom_bed_model(df):
    depts = df['department'].dropna().unique()
    cap = df.groupby('department')['capacity'].first().astype(float).to_dict()
    prob = LpProblem("Custom_Beds", LpMinimize)
    x = {d: LpVariable(f"x_{d}", lowBound=0, upBound=cap[d]) for d in depts}
    s = {d: LpVariable(f"s_{d}", lowBound=0) for d in depts}
    target = {d: 0.95*cap[d] for d in depts}  # example policy target
    prob += lpSum([s[d] for d in depts])
    for d in depts:
        prob += x[d] <= cap[d]
        prob += s[d] >= target[d] - x[d]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
            "allocation": {d: float(x[d].value()) for d in depts},
            "shortages": {d: float(s[d].value()) for d in depts}}"""
        code = st.text_area("Your function: custom_bed_model(df)", value=default_bed_code, height=240, key="bed_custom_code")
        if enable and st.button("▶️ Run Custom Code (Beds)"):
            local_ns = {}
            try:
                exec(code, {"np":np, "pd":pd, "LpProblem":LpProblem, "LpMinimize":LpMinimize, "LpVariable":LpVariable,
                            "lpSum":lpSum, "LpStatus":LpStatus, "value":value, "PULP_CBC_CMD":PULP_CBC_CMD}, local_ns)
                if "custom_bed_model" not in local_ns:
                    st.error("custom_bed_model(df) not defined.")
                else:
                    out = local_ns["custom_bed_model"](bBed)
                    st.session_state["bed_results"] = out
                    st.session_state["bed_filtered"] = bBed.copy()
                    st.success("Custom bed model ran successfully.")
            except Exception as e:
                st.error(f"Error in custom code: {e}")

# -----------------------------
# Tab 2 — Staff
# -----------------------------
with tabs[2]:
    st.markdown('<div class="section-header">👥 Staff Scheduling — Plain-English Problem & LP Model</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<b>Problem in one sentence:</b> Some shifts are short-staffed (costly OT), others are overstaffed (waste).<br><br>
<b>Business impact if we do nothing:</b> higher overtime spend, burnout, turnover, and uneven care quality.<br><br>
<b>What we’re optimizing (LP in simple words):</b> We decide how many staff to assign per role×shift and how much overtime to use so that coverage targets are met at the lowest possible overtime cost.<br><br>
<b>Model we use & how it helps:</b>
<ul>
  <li><b>Min-cost coverage (minimize):</b> picks assignments and overtime so that requirements are met and availability isn’t exceeded. It reveals hotspots where overtime is unavoidable.</li>
</ul>
<b>Constraints (the rules):</b> Assigned + OT ≥ required; Assigned ≤ available; everything ≥ 0.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔎 Filters (Staff)</div>', unsafe_allow_html=True)
    bStaff, sStaff, rStaff = filter_widgets("Staff", bed_df, staff_df, resource_df)

    st.markdown("#### Configure & Run")
    ai_view_s = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], key="ai_staff")
    if st.button("▶️ Run Optimization", key="run_staff"):
        with st.spinner("Solving staffing on current filters…"):
            staff_res = staff_scheduling_basic(sStaff)
            st.session_state["staff_results"] = staff_res
            st.session_state["staff_filtered"] = sStaff.copy()

    if "staff_results" in st.session_state:
        res = st.session_state["staff_results"]
        s_used = st.session_state.get("staff_filtered", sStaff)
        st.markdown('<div class="results">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Solver", res.get("status"))
        with c2: st.metric("Weekly OT Cost", f"${res.get('total_cost',0):,.0f}" if res.get('total_cost') else "—")
        with c3: st.metric("Reported Shortages (30d)", int(s_used['shortage'].sum()) if len(s_used) else 0)
        st.markdown('</div>', unsafe_allow_html=True)

        ov_series = pd.Series(res.get("overtime", {}))
        if not ov_series.empty:
            df_ov = ov_series.rename("overtime").reset_index().rename(columns={"index":"cell"})
            df_ov[['role','shift']] = df_ov['cell'].str.rsplit('_', n=1, expand=True)
            fig = px.bar(df_ov, x="role", y="overtime", color="shift", barmode="group", title="Overtime by Role & Shift")
            fig.update_layout(height=380, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No overtime required under current averages.")

        st.markdown("### AI Summary")
        summary_text = generate_ai_summary("staff_scheduling", res, ai_view_s, llm_model, staff_df=s_used)
        st.markdown(summary_text)

        # Export
        md = f"# Staff Scheduling Report\n\n**Status:** {res.get('status')}\n\n**Weekly OT Cost:** {res.get('total_cost')}\n\n## Overtime Cells\n"
        for k, v in (res.get("overtime",{}) or {}).items():
            if v and v > 0:
                md += f"- {k}: {v:.1f}\n"
        st.download_button("⬇️ Download Staff Report (Markdown)", data=md, file_name="staff_scheduling_report.md", type="secondary")

    with st.expander("📘 LP in 30 seconds + the math we used"):
        st.markdown("""
**Why LP fits staffing:** it trades off overtime vs. availability transparently and proves the minimum possible OT for the current inputs.

**Coverage math:**  
- Variables: assignᵣ,ₛ (regular staff), OTᵣ,ₛ (overtime)  
- Objective: minimize Σ (costᵣ × OTᵣ,ₛ)  
- Constraints: assignᵣ,ₛ + OTᵣ,ₛ ≥ reqᵣ,ₛ; assignᵣ,ₛ ≤ availᵣ,ₛ; all ≥ 0
""")

    with st.expander("🧑‍💻 Advanced — Run Custom Staff Model (unsafe: executes your code)"):
        st.markdown('<span class="small danger">This executes on the server. Use only if you understand the risks.</span>', unsafe_allow_html=True)
        enable = st.checkbox("I understand the risks and want to run custom code", key="staff_custom_enable")
        default_staff_code = """# Define custom_staff_model(df) -> dict like built-in.
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
import pandas as pd
def custom_staff_model(df):
    roles = sorted(df['role'].dropna().unique())
    shifts = sorted(df['shift'].dropna().unique())
    agg = df.groupby(['role','shift']).agg(required=('required','mean'), available=('available','mean'))
    idx = pd.MultiIndex.from_product([roles, shifts], names=['role','shift'])
    agg = agg.reindex(idx).fillna({'required':0.0,'available':0.0})
    req = {(r,s): float(v) for (r,s), v in agg['required'].items()}
    ava = {(r,s): float(v) for (r,s), v in agg['available'].items()}
    cost = {'Nurses':45,'Doctors':80,'Technicians':35,'Support Staff':25}
    prob = LpProblem("Custom_Staff", LpMinimize)
    x = {(r,s): LpVariable(f"x_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    ot = {(r,s): LpVariable(f"ot_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    prob += lpSum([cost.get(r,40)*ot[(r,s)] for r in roles for s in shifts])
    for r in roles:
        for s in shifts:
            prob += x[(r,s)] + ot[(r,s)] >= req[(r,s)]
            prob += x[(r,s)] <= ava[(r,s)]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "total_cost": float(value(prob.objective)) if prob.status == 1 else None,
            "assignments": {f"{r}_{s}": float(x[(r,s)].value()) for r in roles for s in shifts},
            "overtime": {f"{r}_{s}": float(ot[(r,s)].value()) for r in roles for s in shifts},
            "overtime_costs": cost}"""
        code = st.text_area("Your function: custom_staff_model(df)", value=default_staff_code, height=240, key="staff_custom_code")
        if enable and st.button("▶️ Run Custom Code (Staff)"):
            local_ns = {}
            try:
                exec(code, {"pd":pd, "LpProblem":LpProblem, "LpMinimize":LpMinimize, "LpVariable":LpVariable,
                            "lpSum":lpSum, "LpStatus":LpStatus, "value":value, "PULP_CBC_CMD":PULP_CBC_CMD}, local_ns)
                if "custom_staff_model" not in local_ns:
                    st.error("custom_staff_model(df) not defined.")
                else:
                    out = local_ns["custom_staff_model"](sStaff)
                    st.session_state["staff_results"] = out
                    st.session_state["staff_filtered"] = sStaff.copy()
                    st.success("Custom staff model ran successfully.")
            except Exception as e:
                st.error(f"Error in custom code: {e}")

# -----------------------------
# Tab 3 — Resources
# -----------------------------
with tabs[3]:
    st.markdown('<div class="section-header">🧰 Resource Optimization — Plain-English Problem & LP Model</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<b>Problem in one sentence:</b> Some devices sit idle while others are bottlenecks.<br><br>
<b>Business impact if we do nothing:</b> patient delays, longer LOS, and unnecessary capital requests for equipment we don’t truly need.<br><br>
<b>What we’re optimizing (LP in simple words):</b> We choose how many devices of each type to allocate so we maximize usable capacity while respecting maintenance downtime.<br><br>
<b>Model we use & how it helps:</b>
<ul>
  <li><b>Max-utilization (maximize):</b> allocates each device type up to what’s available after maintenance — a clean upper-bound plan that avoids over-promising.</li>
</ul>
<b>Constraints (the rules):</b> Allocation ≤ (total − maintenance); non-negative variables.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔎 Filters (Resources)</div>', unsafe_allow_html=True)
    bRes, sRes, rRes = filter_widgets("Resources", bed_df, staff_df, resource_df)

    st.markdown("#### Configure & Run")
    ai_view_r = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], key="ai_res")
    if st.button("▶️ Run Optimization", key="run_res"):
        with st.spinner("Optimizing resources on current filters…"):
            resR = resource_optimization_basic(rRes)
            st.session_state["res_results"] = resR
            st.session_state["res_filtered"] = rRes.copy()

    base_fig = px.bar(rRes, x='resource', y=['in_use','available','maintenance'],
                      barmode='stack', title="Current Resource Status (Filtered)")
    base_fig.update_layout(height=360, margin=dict(l=8,r=8,t=50,b=8))
    st.plotly_chart(base_fig, use_container_width=True)

    if "res_results" in st.session_state:
        res = st.session_state["res_results"]
        r_used = st.session_state.get("res_filtered", rRes)
        st.markdown('<div class="results">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.metric("Solver", res.get("status"))
        with c2: st.metric("Objective (Units Allocated)", f"{res.get('objective_value',0):.1f}" if res.get('objective_value') else "—")
        st.markdown('</div>', unsafe_allow_html=True)

        alloc_df = pd.DataFrame([(k,v) for k,v in (res.get("allocation",{}) or {}).items()], columns=["Resource","Allocated"])
        if len(alloc_df):
            fig = px.bar(alloc_df, x="Resource", y="Allocated", title="Optimal Resource Allocation")
            fig.update_layout(height=360, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation (check filters and totals).")

        st.markdown("### AI Summary")
        summary_text = generate_ai_summary("resource_optimization", res, ai_view_r, llm_model, resource_df=r_used)
        st.markdown(summary_text)

        # Export
        md = f"# Resource Optimization Report\n\n**Status:** {res.get('status')}\n\n**Objective (Units):** {res.get('objective_value')}\n\n## Allocation\n"
        for k, v in (res.get("allocation",{}) or {}).items():
            md += f"- {k}: {v:.1f}\n"
        st.download_button("⬇️ Download Resource Report (Markdown)", data=md, file_name="resource_optimization_report.md", type="secondary")

    with st.expander("📘 LP in 30 seconds + the math we used"):
        st.markdown("""
**Why LP fits equipment:** it cleanly captures the “can’t allocate more than available after maintenance” rule and gives a best-case throughput plan.

**Utilization math:**  
- Variables: xᵣ = allocation for resource type r  
- Objective: maximize Σ xᵣ  
- Constraint: xᵣ ≤ (totalᵣ − maintenanceᵣ), xᵣ ≥ 0
""")

    with st.expander("🧑‍💻 Advanced — Run Custom Resource Model (unsafe: executes your code)"):
        st.markdown('<span class="small danger">This executes on the server. Use only if you understand the risks.</span>', unsafe_allow_html=True)
        enable = st.checkbox("I understand the risks and want to run custom code", key="res_custom_enable")
        default_res_code = """# Define custom_res_model(df) -> dict like built-ins.
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
def custom_res_model(df):
    resources = df['resource'].dropna().unique()
    total = df.set_index('resource')['total'].astype(float).to_dict()
    maint = df.set_index('resource')['maintenance'].astype(float).to_dict()
    prob = LpProblem("Custom_Resources", LpMaximize)
    x = {r: LpVariable(f"alloc_{r}", lowBound=0, upBound=total[r]) for r in resources}
    prob += lpSum([x[r] for r in resources])
    for r in resources: prob += x[r] <= total[r] - maint[r]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {"status": LpStatus.get(prob.status, str(prob.status)),
            "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
            "allocation": {r: float(x[r].value()) for r in resources}}"""
        code = st.text_area("Your function: custom_res_model(df)", value=default_res_code, height=220, key="res_custom_code")
        if enable and st.button("▶️ Run Custom Code (Resources)"):
            local_ns = {}
            try:
                exec(code, {"LpProblem":LpProblem, "LpMaximize":LpMaximize, "LpVariable":LpVariable,
                            "lpSum":lpSum, "LpStatus":LpStatus, "value":value, "PULP_CBC_CMD":PULP_CBC_CMD}, local_ns)
                if "custom_res_model" not in local_ns:
                    st.error("custom_res_model(df) not defined.")
                else:
                    out = local_ns["custom_res_model"](rRes)
                    st.session_state["res_results"] = out
                    st.session_state["res_filtered"] = rRes.copy()
                    st.success("Custom resource model ran successfully.")
            except Exception as e:
                st.error(f"Error in custom code: {e}")

# -----------------------------
# Tab 4 — Notes & Export
# -----------------------------
with tabs[4]:
    st.markdown("## 🧭 Product & UX Collaboration Notes")
    st.markdown("""
- **Per-tab filters** let executives focus on their unit while analysts isolate cohorts.
- **Executive vs Technical** toggles: outcomes ($, throughput) vs levers (constraints, weights, params).
- **Trust**: Solver status, objective, constraints, and code are visible; AI summaries are generated from filtered data only.
- **Next**: LOS-aware beds, fairness constraints (max consecutive nights, min rest), surge buffers, auto-alerts.
""")

    # Quick combined export (simple)
    if "bed_results" in st.session_state or "staff_results" in st.session_state or "res_results" in st.session_state:
        bundle = "# Hospital Optimization Suite — Snapshot\n\n"
        if "bed_results" in st.session_state:
            res = st.session_state["bed_results"]
            bundle += "## Beds\n"
            bundle += f"- Status: {res.get('status')}\n- Objective: {res.get('objective_value')}\n"
        if "staff_results" in st.session_state:
            res = st.session_state["staff_results"]
            bundle += "\n## Staff\n"
            bundle += f"- Status: {res.get('status')}\n- Weekly OT Cost: {res.get('total_cost')}\n"
        if "res_results" in st.session_state:
            res = st.session_state["res_results"]
            bundle += "\n## Resources\n"
            bundle += f"- Status: {res.get('status')}\n- Objective (Units): {res.get('objective_value')}\n"
        st.download_button("⬇️ Download Suite Snapshot (Markdown)", data=bundle, file_name="hospital_suite_snapshot.md", type="secondary")
