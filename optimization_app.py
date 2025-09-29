# app.py
# Hospital Optimization Suite — v4
# - Robust staffing model (no IndexError)
# - No sidebar, all controls inline
# - Role-aware AI summaries from FILTERED data + model outputs
# - Real filters (Hospital, Department, Admission Type, Condition) if present
# - Editable sections for execs/analysts (notes + code) with export
# - Clear model math + implementation

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Optimization
from pulp import (
    LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
)

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
  .codeblock { background: #0b1021; color: #e5e7eb; padding: 12px; border-radius: 8px; font-size: .85rem; }
  .pill { display:inline-block; padding: 2px 8px; border: 1px solid #cbd5e1; border-radius: 999px; font-size: .75rem; margin-right: 6px; color:#334155;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def normalize_view(v: str) -> str:
    if not v:
        return "executive"
    v = v.lower().strip().replace("-", " ").replace("_", " ")
    if "exec" in v: return "executive"
    if "tech" in v or "deep" in v or "analyst" in v: return "technical"
    return "executive"

@st.cache_data(show_spinner=False)
def try_load_default_csv():
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
                                   "hospital": rng.choice(["North","Central","South"])})
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
    # Make columns robust (lowercase)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Optional columns we’ll use if present
    optional = ["hospital","admission_type","condition"]

    # Try identify bed/staff/resource tables inside one CSV:
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

    for dcol in ["date"]:
        if dcol in bed_df.columns: bed_df[dcol] = pd.to_datetime(bed_df[dcol], errors='coerce')
        if dcol in staff_df.columns: staff_df[dcol] = pd.to_datetime(staff_df[dcol], errors='coerce')
    return bed_df, staff_df, resource_df

def apply_filters(bed_df, staff_df, resource_df,
                  hospital=None, department=None, admission=None, condition=None):
    """Filter all three tables consistently if columns exist."""
    def ftab(tab, col, vals):
        if col in tab.columns and vals:
            return tab[tab[col].isin(vals)]
        return tab

    # Clone to avoid SettingWithCopy
    b, s, r = bed_df.copy(), staff_df.copy(), resource_df.copy()
    b = ftab(b, "hospital", hospital)
    s = ftab(s, "hospital", hospital)
    r = ftab(r, "hospital", hospital)

    b = ftab(b, "department", department)

    b = ftab(b, "admission_type", admission)
    b = ftab(b, "condition", condition)

    return b, s, r

# -----------------------------
# Optimization models (robust)
# -----------------------------
def bed_allocation_basic(bed_df: pd.DataFrame):
    depts = bed_df['department'].dropna().unique()
    if len(depts) == 0:
        return {"status":"Infeasible","objective_value":None,"allocation":{}}

    cap_by = bed_df.groupby('department', dropna=True)['capacity'].first().to_dict()
    prob = LpProblem("Bed_Allocation_Basic", LpMaximize)
    x = {d: LpVariable(f"beds_{d}", lowBound=0, upBound=float(cap_by[d])) for d in depts}

    prob += lpSum([x[d] for d in depts])
    for d in depts:
        prob += x[d] <= float(cap_by[d])

    prob.solve(PULP_CBC_CMD(msg=0))
    return {
        "status": LpStatus.get(prob.status, str(prob.status)),
        "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
        "allocation": {d: float(value(x[d])) for d in depts}
    }

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
    return {
        "status": LpStatus.get(prob.status, str(prob.status)),
        "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
        "allocation": {d: float(value(x[d])) for d in depts},
        "shortages": {d: float(value(s[d])) for d in depts},
        "demand_multipliers": demand_multipliers,
        "weights": weights
    }

def staff_scheduling_basic(staff_df: pd.DataFrame, overtime_costs=None):
    """
    Robust: full role×shift grid; dicts built without iloc (prevents IndexError).
    Works on filtered data as well.
    """
    roles = sorted(staff_df['role'].dropna().unique())
    shifts = sorted(staff_df['shift'].dropna().unique())
    if len(roles) == 0 or len(shifts) == 0:
        return {"status":"Infeasible","total_cost":None,"assignments":{},"overtime":{},"overtime_costs":overtime_costs or {}}

    agg = staff_df.groupby(['role','shift']).agg(
        required=('required','mean'),
        available=('available','mean')
    )

    # Complete grid reindex + fill 0s
    idx = pd.MultiIndex.from_product([roles, shifts], names=['role','shift'])
    agg_full = agg.reindex(idx).fillna({'required':0.0, 'available':0.0})

    # Build dicts from index → values (no .iloc)
    req = {(r,s): float(v) for (r,s), v in agg_full['required'].items()}
    ava = {(r,s): float(v) for (r,s), v in agg_full['available'].items()}

    if overtime_costs is None:
        overtime_costs = {'Nurses': 45, 'Doctors': 80, 'Technicians': 35, 'Support Staff': 25}

    prob = LpProblem("Staff_Scheduling", LpMinimize)
    assign = {(r,s): LpVariable(f"assign_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    ot = {(r,s): LpVariable(f"ot_{r}_{s}", lowBound=0) for r in roles for s in shifts}

    prob += lpSum([overtime_costs.get(r, 40) * ot[(r,s)] for r in roles for s in shifts])

    for r in roles:
        for s in shifts:
            prob += assign[(r,s)] + ot[(r,s)] >= req[(r,s)]
            prob += assign[(r,s)] <= ava[(r,s)]

    prob.solve(PULP_CBC_CMD(msg=0))
    return {
        "status": LpStatus.get(prob.status, str(prob.status)),
        "total_cost": float(value(prob.objective)) if prob.status == 1 else None,
        "assignments": {f"{r}_{s}": float(value(assign[(r,s)])) for r in roles for s in shifts},
        "overtime": {f"{r}_{s}": float(value(ot[(r,s)])) for r in roles for s in shifts},
        "overtime_costs": overtime_costs
    }

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
    return {
        "status": LpStatus.get(prob.status, str(prob.status)),
        "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
        "allocation": {r: float(value(x[r])) for r in resources}
    }

# -----------------------------
# AI summaries (data-driven, filtered)
# -----------------------------
def generate_ai_summary(module, results, view_type, bed_df=None, staff_df=None, resource_df=None):
    vt = normalize_view(view_type)
    def fmt_money(x): return f"${x:,.0f}"
    def pct(x): return f"{x*100:.1f}%"

    if module == "bed_allocation":
        mean_util = None; top_busy = []
        if bed_df is not None and len(bed_df):
            mean_util = bed_df.groupby('department')['utilization_rate'].mean().sort_values(ascending=False)
            top_busy = list(mean_util.index[:2])
        alloc = results.get("allocation", {}) or {}
        total_alloc = sum(alloc.values())
        total_cap = bed_df.groupby('department')['capacity'].first().sum() if (bed_df is not None and len(bed_df)) else None
        alloc_util = (total_alloc/total_cap) if total_cap and total_cap>0 else None
        shortages = results.get("shortages", {}) or {}
        worst_short = sorted(shortages.items(), key=lambda x: x[1], reverse=True)[:2] if shortages else []

        if vt == "executive":
            bullets = []
            if alloc_util is not None:
                bullets.append(f"Projected **system utilization** post-optimization: **{pct(alloc_util)}**.")
            if worst_short:
                bullets.append("Residual risk focus: " + ", ".join([f"{d} ({s:.1f} beds short)" for d,s in worst_short]) + ".")
            if top_busy:
                bullets.append(f"Historic pressure in {', '.join(top_busy)}; model shifts slack from lower-utilized units.")
            # Estimate upside relative to baseline of FILTERED dataset
            baseline = bed_df['utilization_rate'].mean() if (bed_df is not None and len(bed_df)) else 0.0
            eff_gain = max(0.0, (alloc_util or baseline) - baseline)
            est = eff_gain * (total_cap or 0) * 500  # transparent proxy
            if est > 0:
                bullets.append(f"Estimated **annual upside** ≈ {fmt_money(est)} via reduced diversion & higher throughput.")
            recs = [
                "Daily load-balancing target by department (am huddle).",
                "Dynamic bed board with ICU/ED surge rules.",
                "Add LOS predictions to accelerate discharge readiness."
            ]
            return {"title": "🎯 Executive Summary — Bed Allocation", "bullets": bullets, "recs": recs}
        else:
            details = {
                "Solver Status": results.get("status"),
                "Objective": results.get("objective_value"),
                "Demand Multipliers": results.get("demand_multipliers"),
                "Weights": results.get("weights"),
                "Top Shortages": worst_short,
                "Baseline Utilization (filtered)": float(bed_df['utilization_rate'].mean()) if (bed_df is not None and len(bed_df)) else None
            }
            return {"title": "🔬 Technical Deep-Dive — Bed Allocation", "details": details}

    if module == "staff_scheduling":
        ov = results.get("overtime", {}) or {}
        over_items = sorted(ov.items(), key=lambda kv: kv[1], reverse=True)[:3]
        fill_rate = None
        if staff_df is not None and len(staff_df) and staff_df['required'].sum()>0:
            fill_rate = (staff_df['available'].sum()/staff_df['required'].sum())
        if vt == "executive":
            bullets = []
            if fill_rate is not None:
                bullets.append(f"Baseline **fill rate** (filtered): {pct(fill_rate)}.")
            if over_items:
                bullets.append("Overtime hotspots: " + ", ".join([f"{k.replace('_',' ')} ({v:.1f})" for k,v in over_items]))
            tot = results.get("total_cost")
            if tot is not None:
                bullets.append(f"Optimized **weekly overtime cost** ≈ {fmt_money(tot)}.")
            recs = [
                "Night/ED float pool sized for top two hotspots.",
                "Preference bidding to reduce weekend OT.",
                "Rebalance ICU night skill mix vs. acuity."
            ]
            return {"title": "👥 Executive Summary — Staff Scheduling", "bullets": bullets, "recs": recs}
        else:
            details = {
                "Solver Status": results.get("status"),
                "Weekly Overtime Cost": results.get("total_cost"),
                "Top Overtime Cells": over_items,
                "Cost Weights": results.get("overtime_costs")
            }
            return {"title": "📊 Technical Deep-Dive — Staff Scheduling", "details": details}

    if module == "resource_optimization":
        post_rate = None; top = []
        if resource_df is not None and results.get("allocation"):
            alloc = pd.Series(results["allocation"])
            total = resource_df.set_index('resource')['total']
            post_rate = (alloc/total).sort_values(ascending=False)
            top = list(post_rate.index[:3])
        if vt == "executive":
            bullets = []
            if post_rate is not None and len(post_rate):
                bullets.append("Post-optimization utilization leaders: " + ", ".join(top))
                bullets.append(f"Average device utilization after allocation: {pct(post_rate.mean())}")
            recs = [
                "Inter-department swap queue for low-velocity devices.",
                "Maintenance windows outside diagnostic peaks.",
                "RTLS beacons for high-value assets to reduce idle dwell."
            ]
            return {"title": "🔧 Executive Summary — Resource Optimization", "bullets": bullets, "recs": recs}
        else:
            details = {
                "Solver Status": results.get("status"),
                "Objective": results.get("objective_value"),
                "Allocation": results.get("allocation")
            }
            return {"title": "⚙️ Technical Deep-Dive — Resource Optimization", "details": details}

    return {"title": "Summary", "bullets": ["No details."], "recs": []}

# -----------------------------
# Data load
# -----------------------------
default_df = try_load_default_csv()
uploaded = st.file_uploader("📥 Upload CSV (optional) — use your dataset; otherwise a demo dataset is loaded.", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        st.success("File uploaded.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        raw_df = None
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
st.markdown('<div class="subheader">Executives see outcomes. Analysts see levers. Everyone sees the same data.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "🛏️ Beds", "👥 Staff", "🧰 Resources", "🧭 Notes & Code"])

# -----------------------------
# GLOBAL FILTERS (apply to *all* tabs via session)
# -----------------------------
with tabs[0]:
    st.markdown('<div class="section-header">🔎 Global Filters</div>', unsafe_allow_html=True)
    # Build filter choices only from columns that exist
    def choices(tab, col): 
        return sorted([x for x in tab[col].dropna().unique()]) if col in tab.columns else []

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hospitals = st.multiselect("Hospital/Facility", choices(bed_df, "hospital") or choices(staff_df,"hospital") or choices(resource_df,"hospital"))
    with col2:
        departments = st.multiselect("Department", choices(bed_df, "department"))
    with col3:
        admissions = st.multiselect("Admission Type", choices(bed_df, "admission_type"))
    with col4:
        conditions = st.multiselect("Condition", choices(bed_df, "condition"))

    bF, sF, rF = apply_filters(bed_df, staff_df, resource_df,
                               hospital=hospitals, department=departments,
                               admission=admissions, condition=conditions)

    # KPIs (filtered)
    st.markdown('<div class="section-header">📈 Key Performance Indicators (Filtered)</div>', unsafe_allow_html=True)
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

    # Trends
    st.markdown('<div class="section-header">📊 Trends (Filtered)</div>', unsafe_allow_html=True)
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

    # Quick Insights
    st.markdown('<div class="section-header">💡 Quick Insights (Filtered)</div>', unsafe_allow_html=True)
    if len(bF):
        util_by_dept = bF.groupby('department')['utilization_rate'].mean().sort_values(ascending=False)
        top_hot = util_by_dept.head(2).index.tolist()
        low_opps = util_by_dept.tail(1).index.tolist()
        I, J, K = st.columns(3)
        with I:
            st.markdown(f'<div class="ok"><b>Steady Pressure:</b> {", ".join(top_hot) if top_hot else "n/a"}.</div>', unsafe_allow_html=True)
        with J:
            st.markdown(f'<div class="info"><b>Rebalance Opportunity:</b> {", ".join(low_opps) if low_opps else "n/a"}.</div>', unsafe_allow_html=True)
        with K:
            night_short = sF[sF['shift']=="Night"]['shortage'].mean() if len(sF) else 0
            st.markdown(f'<div class="warn"><b>Night Shift:</b> avg shortage ≈ {0 if pd.isna(night_short) else round(night_short,1)}.</div>', unsafe_allow_html=True)
    else:
        st.info("No insights available for current filter.")

# -----------------------------
# Beds
# -----------------------------
with tabs[1]:
    st.markdown('<div class="section-header">🛏️ Bed Allocation — Business Context</div>', unsafe_allow_html=True)

    # Editable problem write-up
    default_bed_problem = ("Suboptimal bed distribution increases wait times and diversion risk. "
                           "Goal: allocate beds to maximize utilization or minimize weighted shortages, per capacity and surge needs.")
    st.session_state.setdefault("bed_problem", default_bed_problem)
    st.session_state["bed_problem"] = st.text_area("Problem statement (editable)", st.session_state["bed_problem"])

    # Controls (per filtered dataset)
    st.markdown("#### Configure & Run")
    model = st.radio("Optimization approach", ["Basic Utilization", "Demand-Based (weighted shortages)"], horizontal=True, key="bed_model")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        ai_view_b = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], horizontal=True, key="ai_b")
    with col2:
        demand_stress = st.slider("Demand stress test (+/- %)", -20, 40, 0)
    with col3:
        icu_bump = st.slider("ICU priority bump", 0, 5, 2)

    run_beds = st.button("🚀 Run Bed Optimization", type="primary")
    # Use filtered tables from Overview tab: re-apply to be safe
    bF2, _, _ = apply_filters(bed_df, staff_df, resource_df,
                              hospital=st.session_state.get("hospital_sel"),
                              department=st.session_state.get("dept_sel"),
                              admission=st.session_state.get("admission_sel"),
                              condition=st.session_state.get("condition_sel"))

    # If we didn't store those in session, use current Overview filters directly
    # (we stored nothing explicit; reuse local variables since tabs share state within run)
    bF2 = bF if 'bF' in locals() else bed_df

    if run_beds:
        with st.spinner("Optimizing bed allocation (on filtered data)…"):
            if model.startswith("Basic"):
                bed_res = bed_allocation_basic(bF2)
            else:
                util = bF2.groupby('department')['utilization_rate'].mean() if len(bF2) else pd.Series(dtype=float)
                dm = {d: float(np.clip((u if not np.isnan(u) else 0.8) * (1 + demand_stress/100.0), 0.8, 1.4)) for d,u in util.items()}
                weights = {d: (10 + icu_bump if d=="ICU" else (9 if d=="Emergency" else 6)) for d in util.index}
                bed_res = bed_allocation_demand_based(bF2, dm if dm else None, weights if weights else None)
            st.session_state["bed_results"] = bed_res
            st.session_state["bed_filtered"] = bF2.copy()

    if "bed_results" in st.session_state:
        res = st.session_state["bed_results"]
        b_used = st.session_state.get("bed_filtered", bF2)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Solver", res.get("status","?"))
        with c2:
            obj = res.get("objective_value")
            st.metric("Objective", f"{obj:.2f}" if obj is not None else "—")
        with c3: st.metric("Departments", b_used['department'].nunique() if len(b_used) else 0)
        with c4: st.metric("Total Capacity", int(b_used.groupby('department')['capacity'].first().sum()) if len(b_used) else 0)

        alloc_df = pd.DataFrame([(k,v) for k,v in (res.get("allocation",{}) or {}).items()], columns=["Department","Allocated Beds"])
        if len(alloc_df):
            fig = px.bar(alloc_df, x="Department", y="Allocated Beds", title="Optimal Bed Allocation")
            fig.update_layout(height=380, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation (check filters and capacities).")

        summary = generate_ai_summary("bed_allocation", res, ai_view_b, bed_df=b_used)
        if normalize_view(ai_view_b) == "executive":
            st.markdown("### Executive Summary (from filtered data & model outputs)")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")
            st.markdown("**Recommendations:**")
            for r in summary["recs"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("### Technical Deep-Dive")
            st.json(summary["details"])

    with st.expander("📘 Model Math & Implementation (editable notes + copyable code)"):
        math_text = ("**Basic Utilization (LP)**\n\n"
                     "Maximize  Σ xᵢ\n\n"
                     "Subject to: 0 ≤ xᵢ ≤ capacityᵢ\n\n"
                     "**Demand-Based (LP)**\n\n"
                     "Minimize  Σ wᵢ·sᵢ\n\n"
                     "Subject to: xᵢ ≤ capacityᵢ,  sᵢ ≥ demandᵢ − xᵢ,  xᵢ, sᵢ ≥ 0")
        st.markdown(math_text)

        bed_code = """def bed_allocation_demand_based(bed_df, demand_multipliers=None, weights=None):
    depts = bed_df['department'].dropna().unique()
    cap = bed_df.groupby('department', dropna=True)['capacity'].first().astype(float).to_dict()

    # Defaults from data when not provided
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
            "demand_multipliers": demand_multipliers, "weights": weights}"""
        custom_bed_code = st.text_area("Editable notes / code (for your doc or PRs)", bed_code, height=220)
        st.download_button("⬇️ Download Bed Model Snippet", data=custom_bed_code.encode("utf-8"),
                           file_name="bed_allocation_model.py", type="secondary")

# -----------------------------
# Staff
# -----------------------------
with tabs[2]:
    st.markdown('<div class="section-header">👥 Staff Scheduling — Business Context</div>', unsafe_allow_html=True)

    default_staff_problem = ("Coverage gaps and overtime cost degrade care quality and retention. "
                             "Goal: minimize overtime cost subject to minimum requirements and availability.")
    st.session_state.setdefault("staff_problem", default_staff_problem)
    st.session_state["staff_problem"] = st.text_area("Problem statement (editable)", st.session_state["staff_problem"])

    st.markdown("#### Configure & Run (uses filtered staff table)")
    ai_view_s = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], horizontal=True, key="ai_staff")
    run_staff = st.button("🚀 Optimize Staff Schedules", type="primary")

    # Use filtered staff table from Overview
    sF2 = sF if 'sF' in locals() else staff_df

    if run_staff:
        with st.spinner("Solving staffing (on filtered data)…"):
            staff_res = staff_scheduling_basic(sF2)
            st.session_state["staff_results"] = staff_res
            st.session_state["staff_filtered"] = sF2.copy()

    if "staff_results" in st.session_state:
        res = st.session_state["staff_results"]
        s_used = st.session_state.get("staff_filtered", sF2)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Solver", res.get("status"))
        with c2: st.metric("Weekly OT Cost", f"${res.get('total_cost',0):,.0f}" if res.get('total_cost') else "—")
        with c3: st.metric("Reported Shortages (30d)", int(s_used['shortage'].sum()) if len(s_used) else 0)

        ov_series = pd.Series(res.get("overtime", {}))
        if not ov_series.empty:
            df_ov = ov_series.rename("overtime").reset_index().rename(columns={"index":"cell"})
            df_ov[['role','shift']] = df_ov['cell'].str.rsplit('_', n=1, expand=True)
            fig = px.bar(df_ov, x="role", y="overtime", color="shift", barmode="group", title="Overtime by Role & Shift")
            fig.update_layout(height=380, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No overtime required under current averages.")

        summary = generate_ai_summary("staff_scheduling", res, ai_view_s, staff_df=s_used)
        if normalize_view(ai_view_s) == "executive":
            st.markdown("### Executive Summary (from filtered data & model outputs)")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")
            st.markdown("**Recommendations:**")
            for r in summary["recs"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("### Technical Deep-Dive")
            st.json(summary["details"])

    with st.expander("📘 Model Math & Implementation (editable notes + copyable code)"):
        staff_code = """def staff_scheduling_basic(staff_df, overtime_costs=None):
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
            "overtime_costs": overtime_costs}"""
        custom_staff_code = st.text_area("Editable notes / code (for your doc or PRs)", staff_code, height=260)
        st.download_button("⬇️ Download Staff Model Snippet", data=custom_staff_code.encode("utf-8"),
                           file_name="staff_scheduling_model.py", type="secondary")

# -----------------------------
# Resources
# -----------------------------
with tabs[3]:
    st.markdown('<div class="section-header">🧰 Resource Optimization — Business Context</div>', unsafe_allow_html=True)

    default_res_problem = ("Uneven device utilization creates bottlenecks and drives unnecessary capex. "
                           "Goal: maximize usable allocation subject to maintenance.")
    st.session_state.setdefault("res_problem", default_res_problem)
    st.session_state["res_problem"] = st.text_area("Problem statement (editable)", st.session_state["res_problem"])

    st.markdown("#### Configure & Run (uses filtered resources table)")
    ai_view_r = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], horizontal=True, key="ai_res")
    run_res = st.button("🚀 Optimize Resources", type="primary")

    rF2 = rF if 'rF' in locals() else resource_df
    base_fig = px.bar(rF2, x='resource', y=['in_use','available','maintenance'], barmode='stack', title="Current Resource Status (Filtered)")
    base_fig.update_layout(height=360, margin=dict(l=8,r=8,t=50,b=8))
    st.plotly_chart(base_fig, use_container_width=True)

    if run_res:
        with st.spinner("Optimizing resources (on filtered data)…"):
            resR = resource_optimization_basic(rF2)
            st.session_state["res_results"] = resR
            st.session_state["res_filtered"] = rF2.copy()

    if "res_results" in st.session_state:
        res = st.session_state["res_results"]
        c1, c2 = st.columns(2)
        with c1: st.metric("Solver", res.get("status"))
        with c2: st.metric("Objective (Units Allocated)", f"{res.get('objective_value',0):.1f}" if res.get('objective_value') else "—")

        alloc_df = pd.DataFrame([(k,v) for k,v in (res.get("allocation",{}) or {}).items()], columns=["Resource","Allocated"])
        if len(alloc_df):
            fig = px.bar(alloc_df, x="Resource", y="Allocated", title="Optimal Resource Allocation")
            fig.update_layout(height=360, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation (check filters and totals).")

        summary = generate_ai_summary("resource_optimization", res, ai_view_r, resource_df=st.session_state.get("res_filtered", rF2))
        if normalize_view(ai_view_r) == "executive":
            st.markdown("### Executive Summary (from filtered data & model outputs)")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")
            st.markdown("**Recommendations:**")
            for r in summary["recs"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("### Technical Deep-Dive")
            st.json(summary["details"])

    with st.expander("📘 Model Math & Implementation (editable notes + copyable code)"):
        res_code = """def resource_optimization_basic(resource_df):
    resources = resource_df['resource'].dropna().unique()
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
            "allocation": {r: float(value(x[r])) for r in resources}}"""
        custom_res_code = st.text_area("Editable notes / code (for your doc or PRs)", res_code, height=200)
        st.download_button("⬇️ Download Resource Model Snippet", data=custom_res_code.encode("utf-8"),
                           file_name="resource_optimization_model.py", type="secondary")

# -----------------------------
# Notes & Code (product + UX + exports)
# -----------------------------
with tabs[4]:
    st.markdown("## 🧭 Product & UX Collaboration Notes (Manisha Arora × UX)")
    st.markdown("""
- **Audience modes:** Prominent exec/technical toggles; execs see $ and throughput; analysts see constraints, weights, and data lineage.
- **No side panes:** All controls live by their visuals; fewer clicks, clearer mental model.
- **Trust:** Solver status, objective, constraints are transparent; code snippets match the running implementation.
- **Scenario controls:** Only high-impact sliders (demand stress, ICU weight) + real filters (Hospital/Dept/Admission/Condition).
- **Next:** LOS-aware beds, fairness (max consecutive nights, 12-hr rest), swap queue, alerting on thresholds.
""")

    # Export quick snapshots if present
    if "bed_results" in st.session_state:
        res = st.session_state["bed_results"]
        b_used = st.session_state.get("bed_filtered")
        md = f"# Bed Allocation Report\n\n**Status:** {res.get('status')}\n\n**Objective:** {res.get('objective_value')}\n\n## Allocation\n"
        for d, v in (res.get("allocation",{}) or {}).items():
            md += f"- {d}: {v:.1f}\n"
        if b_used is not None and len(b_used):
            md += "\n## Filter Context\n"
            for col in ["hospital","department","admission_type","condition"]:
                if col in b_used.columns:
                    vals = ", ".join(sorted([str(x) for x in b_used[col].dropna().unique()]))
                    md += f"- {col}: {vals}\n"
        st.download_button("⬇️ Download Bed Report (MD)", data=md, file_name="bed_allocation_report.md", type="secondary")

