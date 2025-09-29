# app.py
# Hospital Optimization Suite — v3 (no sidebar, exec/analyst friendly, hardened models)

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Optimization
from pulp import (
    LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpStatus,
    value, PULP_CBC_CMD
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
    .main-header { font-size: 2.2rem; color: #0f172a; font-weight: 800; margin: 0 0 .25rem 0;}
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
                         "utilization_rate": occ/cap})
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
                                   "shortage": max(0, req - int(avail))})
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
                         "utilization_rate": in_use/total})
    resource_df = pd.DataFrame(res_rows)
    return bed_df, staff_df, resource_df

def split_or_infer(df: pd.DataFrame):
    bed_cols = {"date","department","capacity","occupied","available","utilization_rate"}
    staff_cols = {"date","role","shift","required","available","shortage"}
    res_cols = {"resource","total","in_use","maintenance","available","utilization_rate"}

    df = df.rename(columns={c: c.lower() for c in df.columns})
    cols = set(df.columns)

    bed_df = df[list(bed_cols)] if bed_cols.issubset(cols) else None
    staff_df = df[list(staff_cols)] if staff_cols.issubset(cols) else None
    resource_df = df[list(res_cols)] if res_cols.issubset(cols) else None

    if bed_df is None or staff_df is None or resource_df is None:
        sb, ss, sr = generate_synthetic()
        bed_df = bed_df if bed_df is not None else sb
        staff_df = staff_df if staff_df is not None else ss
        resource_df = resource_df if resource_df is not None else sr

    if "date" in bed_df.columns: bed_df["date"] = pd.to_datetime(bed_df["date"])
    if "date" in staff_df.columns: staff_df["date"] = pd.to_datetime(staff_df["date"])
    return bed_df, staff_df, resource_df

# -----------------------------
# Optimization models
# -----------------------------
def bed_allocation_basic(bed_df: pd.DataFrame):
    departments = bed_df['department'].unique()
    cap_by_dept = bed_df.groupby('department')['capacity'].first().to_dict()

    prob = LpProblem("Bed_Allocation_Basic", LpMaximize)
    x = {d: LpVariable(f"beds_{d}", lowBound=0, upBound=cap_by_dept[d]) for d in departments}

    # Objective: maximize allocated beds
    prob += lpSum([x[d] for d in departments])

    # (upBound already enforces capacity)
    for d in departments:
        prob += x[d] <= cap_by_dept[d]

    prob.solve(PULP_CBC_CMD(msg=0))
    return {
        "status": LpStatus[prob.status],
        "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
        "allocation": {d: float(value(x[d])) for d in departments}
    }

def bed_allocation_demand_based(bed_df: pd.DataFrame, demand_multipliers=None, weights=None):
    departments = bed_df['department'].unique()
    cap = bed_df.groupby('department')['capacity'].first().to_dict()

    if demand_multipliers is None:
        util = bed_df.groupby('department')['utilization_rate'].mean().to_dict()
        demand_multipliers = {d: float(np.clip(util.get(d, 0.8)*1.05 + 0.05, 0.85, 1.35)) for d in departments}
    if weights is None:
        weights = {d: (10 if d in ("ICU","Emergency") else 6) for d in departments}

    prob = LpProblem("Bed_Allocation_Demand", LpMinimize)
    x = {d: LpVariable(f"beds_{d}", lowBound=0, upBound=cap[d]) for d in departments}
    s = {d: LpVariable(f"short_{d}", lowBound=0) for d in departments}

    # Minimize weighted shortages
    prob += lpSum([weights[d]*s[d] for d in departments])

    for d in departments:
        expected = cap[d] * demand_multipliers[d]
        prob += x[d] <= cap[d]
        prob += s[d] >= expected - x[d]

    prob.solve(PULP_CBC_CMD(msg=0))
    return {
        "status": LpStatus[prob.status],
        "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
        "allocation": {d: float(value(x[d])) for d in departments},
        "shortages": {d: float(value(s[d])) for d in departments},
        "demand_multipliers": demand_multipliers,
        "weights": weights
    }

def staff_scheduling_basic(staff_df: pd.DataFrame, overtime_costs=None):
    """
    Robust version: builds full cartesian grid of role×shift, fills missing with 0 to avoid IndexError.
    """
    roles = sorted(staff_df['role'].dropna().unique())
    shifts = sorted(staff_df['shift'].dropna().unique())

    agg = staff_df.groupby(['role','shift']).agg(
        required=('required','mean'),
        available=('available','mean')
    ).reset_index()

    # Build complete grid
    idx = pd.MultiIndex.from_product([roles, shifts], names=['role','shift'])
    agg_full = (agg.set_index(['role','shift'])
                   .reindex(idx)
                   .fillna({'required': 0.0, 'available': 0.0})
                   .reset_index())

    req = {(r,s): float(agg_full.loc[(agg_full.role==r) & (agg_full.shift==s), 'required'].iloc[0])
           for r in roles for s in shifts}
    ava = {(r,s): float(agg_full.loc[(agg_full.role==r) & (agg_full.shift==s), 'available'].iloc[0])
           for r in roles for s in shifts}

    if overtime_costs is None:
        overtime_costs = {'Nurses': 45, 'Doctors': 80, 'Technicians': 35, 'Support Staff': 25}

    prob = LpProblem("Staff_Scheduling", LpMinimize)
    assign = {(r,s): LpVariable(f"assign_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    ot = {(r,s): LpVariable(f"ot_{r}_{s}", lowBound=0) for r in roles for s in shifts}

    # Objective: minimize overtime cost
    prob += lpSum([overtime_costs.get(r, 40) * ot[(r,s)] for r in roles for s in shifts])

    for r in roles:
        for s in shifts:
            prob += assign[(r,s)] + ot[(r,s)] >= req[(r,s)]
            prob += assign[(r,s)] <= ava[(r,s)]

    prob.solve(PULP_CBC_CMD(msg=0))
    return {
        "status": LpStatus[prob.status],
        "total_cost": float(value(prob.objective)) if prob.status == 1 else None,
        "assignments": {f"{r}_{s}": float(value(assign[(r,s)])) for r in roles for s in shifts},
        "overtime": {f"{r}_{s}": float(value(ot[(r,s)])) for r in roles for s in shifts},
        "overtime_costs": overtime_costs
    }

def resource_optimization_basic(resource_df: pd.DataFrame):
    resources = resource_df['resource'].unique()
    total = resource_df.set_index('resource')['total'].to_dict()
    maint = resource_df.set_index('resource')['maintenance'].to_dict()

    prob = LpProblem("Resource_Optimization", LpMaximize)
    x = {r: LpVariable(f"alloc_{r}", lowBound=0, upBound=total[r]) for r in resources}
    prob += lpSum([x[r] for r in resources])

    for r in resources:
        prob += x[r] <= total[r] - maint[r]

    prob.solve(PULP_CBC_CMD(msg=0))
    return {
        "status": LpStatus[prob.status],
        "objective_value": float(value(prob.objective)) if prob.status == 1 else None,
        "allocation": {r: float(value(x[r])) for r in resources}
    }

# -----------------------------
# AI summaries (data-driven)
# -----------------------------
def generate_ai_summary(module, results, view_type, bed_df=None, staff_df=None, resource_df=None):
    vt = normalize_view(view_type)
    def fmt_money(x): return f"${x:,.0f}"
    def pct(x): return f"{x*100:.1f}%"

    if module == "bed_allocation":
        mean_util = bed_df.groupby('department')['utilization_rate'].mean().sort_values(ascending=False) if bed_df is not None else None
        top_busy = list(mean_util.index[:2]) if mean_util is not None else []
        alloc = results.get("allocation", {})
        total_alloc = sum(alloc.values()) if alloc else 0
        total_cap = bed_df.groupby('department')['capacity'].first().sum() if bed_df is not None else None
        alloc_util = total_alloc/total_cap if total_cap else None
        shortages = results.get("shortages", {})
        worst_short = sorted(shortages.items(), key=lambda x: x[1], reverse=True)[:2] if shortages else []

        if vt == "executive":
            bullets = []
            if alloc_util is not None:
                bullets.append(f"Projected **system utilization** post-optimization: **{pct(alloc_util)}**.")
            if worst_short:
                bullets.append("Residual risk focus: " + ", ".join([f"{d} ({s:.1f} beds short)" for d,s in worst_short]) + ".")
            if top_busy:
                bullets.append(f"Historic pressure persists in {', '.join(top_busy)}; model shifts slack from lower-utilized units.")
            baseline = bed_df['utilization_rate'].mean() if bed_df is not None else 0.78
            eff_gain = max(0.0, (alloc_util or baseline) - baseline)
            est = (eff_gain * (total_cap or 0) * 500)
            bullets.append(f"Estimated **annual upside** ≈ {fmt_money(est)} via reduced diversion and higher throughput.")
            recs = [
                "Stand up a daily load-balancing huddle with department-specific targets.",
                "Enable a dynamic bed board with ICU/Emergency surge rules.",
                "Add LOS predictions to accelerate discharge readiness."
            ]
            return {"title": "🎯 Executive Summary — Bed Allocation", "bullets": bullets, "recs": recs}
        else:
            details = {
                "Solver Status": results.get("status"),
                "Objective": results.get("objective_value"),
                "Demand Multipliers": results.get("demand_multipliers"),
                "Weights": results.get("weights"),
                "Top Shortages": worst_short
            }
            return {"title": "🔬 Technical Deep-Dive — Bed Allocation", "details": details}

    if module == "staff_scheduling":
        ov = results.get("overtime", {}) or {}
        over_items = sorted(ov.items(), key=lambda kv: kv[1], reverse=True)[:3]
        fill_rate = (staff_df['available'].sum()/staff_df['required'].sum()) if staff_df is not None else None
        if vt == "executive":
            bullets = []
            if fill_rate is not None:
                bullets.append(f"Baseline **fill rate**: {pct(fill_rate)}.")
            if over_items:
                bullets.append("Overtime hotspots: " + ", ".join([f"{k.replace('_',' ')} ({v:.1f})" for k,v in over_items]))
            tot = results.get("total_cost")
            if tot is not None:
                bullets.append(f"Optimized **weekly overtime cost** ≈ {fmt_money(tot)}.")
            recs = [
                "Create a cross-trained Night/ED float pool.",
                "Introduce preference bidding to reduce involuntary weekend OT.",
                "Align skill-mix with ICU night acuity; rebalance certifications."
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
        if resource_df is not None and results.get("allocation"):
            alloc = pd.Series(results["allocation"])
            total = resource_df.set_index('resource')['total']
            post_rate = (alloc/total).sort_values(ascending=False)
            top = list(post_rate.index[:3])
        else:
            post_rate, top = None, []
        if vt == "executive":
            bullets = []
            if post_rate is not None:
                bullets.append("Post-optimization utilization leaders: " + ", ".join(top))
                bullets.append(f"Average device utilization after allocation: {pct(post_rate.mean())}")
            recs = [
                "Stand up an inter-department swap queue for low-velocity devices.",
                "Schedule maintenance outside diagnostic peaks.",
                "Add RTLS beacons to high-value assets to shrink idle time."
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
# Data loading
# -----------------------------
default_df = try_load_default_csv()
uploaded = st.file_uploader("📥 Upload CSV (optional) — if omitted, we'll use your default dataset or a synthetic demo.", type=["csv"])
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
# Header + Nav
# -----------------------------
st.markdown('<div class="main-header">🏥 Hospital Optimization Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Clear for execs, deep for analysts — with real, data-driven insights.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "🛏️ Beds", "👥 Staff", "🧰 Resources", "🧭 Report & Design Notes"])

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_u = bed_df['utilization_rate'].mean()
        st.markdown(f'<div class="kpi"><div class="muted">Bed Utilization</div><h3>{avg_u:.1%}</h3><div class="muted">Avg across depts</div></div>', unsafe_allow_html=True)
    with c2:
        staff_short = int(staff_df['shortage'].sum())
        st.markdown(f'<div class="kpi"><div class="muted">Staff Shortage</div><h3>{staff_short}</h3><div class="muted">Shifts understaffed (30d)</div></div>', unsafe_allow_html=True)
    with c3:
        avg_res = resource_df['utilization_rate'].mean()
        st.markdown(f'<div class="kpi"><div class="muted">Resource Utilization</div><h3>{avg_res:.1%}</h3><div class="muted">Avg devices in use</div></div>', unsafe_allow_html=True)
    with c4:
        potential = max(0, (avg_u-0.7)) * bed_df.groupby('department')['capacity'].first().sum() * 600
        st.markdown(f'<div class="kpi"><div class="muted">Potential Upside</div><h3>${potential:,.0f}</h3><div class="muted">Throughput & overtime</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Trends</div>', unsafe_allow_html=True)
    A, B = st.columns(2)
    with A:
        daily = bed_df.groupby(['date','department'])['utilization_rate'].mean().reset_index()
        fig = px.line(daily, x='date', y='utilization_rate', color='department', title="Bed Utilization by Department (90d)")
        fig.update_layout(yaxis_tickformat='.0%', height=360, margin=dict(l=8,r=8,t=50,b=8))
        st.plotly_chart(fig, use_container_width=True)
    with B:
        res_fig = go.Figure()
        res_fig.add_trace(go.Bar(name='In Use', x=resource_df['resource'], y=resource_df['in_use']))
        res_fig.add_trace(go.Bar(name='Available', x=resource_df['resource'], y=resource_df['available']))
        res_fig.add_trace(go.Bar(name='Maintenance', x=resource_df['resource'], y=resource_df['maintenance']))
        res_fig.update_layout(barmode='stack', title="Resource Status", height=360, margin=dict(l=8,r=8,t=50,b=8))
        st.plotly_chart(res_fig, use_container_width=True)

    st.markdown('<div class="section-header">💡 Quick Insights</div>', unsafe_allow_html=True)
    util_by_dept = bed_df.groupby('department')['utilization_rate'].mean().sort_values(ascending=False)
    top_hot = util_by_dept.head(2).index.tolist()
    low_opps = util_by_dept.tail(1).index.tolist()
    I, J, K = st.columns(3)
    with I:
        st.markdown(f'<div class="ok"><b>Steady Pressure:</b> {", ".join(top_hot)} show sustained demand—watch diversion risk on peaks.</div>', unsafe_allow_html=True)
    with J:
        st.markdown(f'<div class="info"><b>Rebalance Opportunity:</b> {", ".join(low_opps)} can lend slack during daytime.</div>', unsafe_allow_html=True)
    with K:
        night_short = staff_df[staff_df['shift']=="Night"]['shortage'].mean()
        st.markdown(f'<div class="warn"><b>Night Shift:</b> average shortage ≈ {night_short:.1f}. Consider a targeted float pool.</div>', unsafe_allow_html=True)

# -----------------------------
# Beds
# -----------------------------
with tabs[1]:
    st.markdown('<div class="section-header">🛏️ Bed Allocation — Business Context</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<b>Problem:</b> Suboptimal bed distribution drives wait times, diversion, and revenue loss.<br>
<b>Goal:</b> Allocate beds to maximize utilization or minimize weighted shortages while respecting capacity.<br>
<b>Who cares:</b> Executives (throughput, diversion) • Bed managers (daily operations) • Analysts (model stability).
</div>
""", unsafe_allow_html=True)

    # Model choice + controls in-page
    model = st.radio("Choose optimization approach", ["Basic Utilization", "Demand-Based (weighted shortages)"], horizontal=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        ai_view = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], horizontal=True)
    with col2:
        demand_stress = st.slider("Demand stress test (+/- %)", -20, 40, 0)
    with col3:
        icu_bump = st.slider("ICU priority bump", 0, 5, 2)
    run_beds = st.button("🚀 Run Bed Optimization", type="primary")

    if run_beds:
        with st.spinner("Optimizing bed allocation…"):
            if model.startswith("Basic"):
                bed_res = bed_allocation_basic(bed_df)
            else:
                util = bed_df.groupby('department')['utilization_rate'].mean()
                dm = {d: float(np.clip(u * (1 + demand_stress/100.0), 0.8, 1.4)) for d,u in util.items()}
                weights = {d: (10 + icu_bump if d=="ICU" else (9 if d=="Emergency" else 6)) for d in util.index}
                bed_res = bed_allocation_demand_based(bed_df, dm, weights)
            st.session_state["bed_results"] = bed_res

    if "bed_results" in st.session_state:
        res = st.session_state["bed_results"]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Solver", res.get("status","?"))
        with c2:
            obj = res.get("objective_value")
            st.metric("Objective", f"{obj:.2f}" if obj is not None else "—")
        with c3: st.metric("Departments", bed_df['department'].nunique())
        with c4: st.metric("Total Capacity", int(bed_df.groupby('department')['capacity'].first().sum()))
        alloc_df = pd.DataFrame([(k,v) for k,v in res.get("allocation",{}).items()], columns=["Department","Allocated Beds"])
        fig = px.bar(alloc_df, x="Department", y="Allocated Beds", title="Optimal Bed Allocation")
        fig.update_layout(height=380, margin=dict(l=8,r=8,t=50,b=8))
        st.plotly_chart(fig, use_container_width=True)

        summary = generate_ai_summary("bed_allocation", res, ai_view, bed_df=bed_df)
        if normalize_view(ai_view) == "executive":
            st.markdown("### Executive Summary")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")
            st.markdown("**Recommendations:**")
            for r in summary["recs"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("### Technical Deep-Dive")
            st.json(summary["details"])

    with st.expander("📘 Model Details & Implementation"):
        st.markdown("**Basic Utilization (LP)**  \nMaximize Σ xᵢ subject to 0 ≤ xᵢ ≤ capacityᵢ.")
        st.markdown("**Demand-Based (LP)**  \nMinimize Σ wᵢ·sᵢ subject to xᵢ ≤ capacityᵢ and sᵢ ≥ demandᵢ − xᵢ.")
        st.code(
            """def bed_allocation_demand_based(bed_df, demand_multipliers, weights):
    departments = bed_df['department'].unique()
    cap = bed_df.groupby('department')['capacity'].first().to_dict()
    prob = LpProblem("Bed_Allocation_Demand", LpMinimize)
    x = {d: LpVariable(f"beds_{d}", lowBound=0, upBound=cap[d]) for d in departments}
    s = {d: LpVariable(f"short_{d}", lowBound=0) for d in departments}
    prob += lpSum([weights[d]*s[d] for d in departments])
    for d in departments:
        expected = cap[d] * demand_multipliers[d]
        prob += x[d] <= cap[d]
        prob += s[d] >= expected - x[d]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {...}""",
            language="python"
        )

# -----------------------------
# Staff
# -----------------------------
with tabs[2]:
    st.markdown('<div class="section-header">👥 Staff Scheduling — Business Context</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<b>Problem:</b> Coverage gaps and costly overtime degrade care and staff well-being.<br>
<b>Goal:</b> Minimize overtime cost while meeting minimum requirements and respecting availability.<br>
<b>Who cares:</b> Execs (labor $), schedulers (operability), analysts (fairness & constraints).
</div>
""", unsafe_allow_html=True)

    L, R = st.columns([1,1])
    with L:
        ai_view_s = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], horizontal=True, key="ai_staff")
    with R:
        weekend_premium = st.slider("Weekend premium (not applied in demo model)", 0, 50, 20)
    run_staff = st.button("🚀 Optimize Staff Schedules", type="primary")

    if run_staff:
        with st.spinner("Solving scheduling…"):
            staff_res = staff_scheduling_basic(staff_df)
            st.session_state["staff_results"] = staff_res

    if "staff_results" in st.session_state:
        res = st.session_state["staff_results"]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Solver", res.get("status"))
        with c2: st.metric("Weekly OT Cost", f"${res.get('total_cost',0):,.0f}" if res.get('total_cost') else "—")
        with c3: st.metric("Reported Shortages (30d)", int(staff_df['shortage'].sum()))
        ov = pd.Series(res.get("overtime", {}))
        if not ov.empty:
            df_ov = ov.rename("overtime").reset_index().rename(columns={"index":"cell"})
            # split at last underscore (role may include spaces; only separator is one underscore)
            df_ov[['role','shift']] = df_ov['cell'].str.rsplit('_', n=1, expand=True)
            fig = px.bar(df_ov, x="role", y="overtime", color="shift",
                         barmode="group", title="Overtime by Role & Shift")
            fig.update_layout(height=380, margin=dict(l=8,r=8,t=50,b=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No overtime required under current averages.")

        summary = generate_ai_summary("staff_scheduling", res, ai_view_s, staff_df=staff_df)
        if normalize_view(ai_view_s) == "executive":
            st.markdown("### Executive Summary")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")
            st.markdown("**Recommendations:**")
            for r in summary["recs"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("### Technical Deep-Dive")
            st.json(summary["details"])

    with st.expander("📘 Model Details & Implementation"):
        st.markdown("**Coverage LP (demo):** Minimize Σ costᵣ·OTᵣ,ₛ s.t. assignᵣ,ₛ + OTᵣ,ₛ ≥ reqᵣ,ₛ and assignᵣ,ₛ ≤ availᵣ,ₛ.")
        st.code(
            """def staff_scheduling_basic(staff_df, overtime_costs=None):
    roles = sorted(staff_df['role'].dropna().unique())
    shifts = sorted(staff_df['shift'].dropna().unique())
    agg = staff_df.groupby(['role','shift']).agg(required=('required','mean'), available=('available','mean')).reset_index()
    idx = pd.MultiIndex.from_product([roles, shifts], names=['role','shift'])
    agg_full = (agg.set_index(['role','shift']).reindex(idx).fillna({'required':0.0,'available':0.0}).reset_index())
    req = {(r,s): float(agg_full[(agg_full.role==r)&(agg_full.shift==s)]['required'].iloc[0]) for r in roles for s in shifts}
    ava = {(r,s): float(agg_full[(agg_full.role==r)&(agg_full.shift==s)]['available'].iloc[0]) for r in roles for s in shifts}
    if overtime_costs is None: overtime_costs={'Nurses':45,'Doctors':80,'Technicians':35,'Support Staff':25}
    prob = LpProblem("Staff_Scheduling", LpMinimize)
    assign = {(r,s): LpVariable(f"assign_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    ot = {(r,s): LpVariable(f"ot_{r}_{s}", lowBound=0) for r in roles for s in shifts}
    prob += lpSum([overtime_costs.get(r,40)*ot[(r,s)] for r in roles for s in shifts])
    for r in roles:
        for s in shifts:
            prob += assign[(r,s)] + ot[(r,s)] >= req[(r,s)]
            prob += assign[(r,s)] <= ava[(r,s)]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {...}""",
            language="python"
        )

# -----------------------------
# Resources
# -----------------------------
with tabs[3]:
    st.markdown('<div class="section-header">🧰 Resource Optimization — Business Context</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card">
<b>Problem:</b> Under/over-utilized equipment causes bottlenecks and capex pressure.<br>
<b>Goal:</b> Maximize usable allocation subject to maintenance constraints.<br>
<b>Who cares:</b> Execs (ROI), biomed/ops (availability), analysts (trade-offs).
</div>
""", unsafe_allow_html=True)

    ai_view_r = st.radio("Analysis View", ["Executive Summary", "Technical Deep-Dive"], horizontal=True, key="ai_res")
    run_res = st.button("🚀 Optimize Resources", type="primary")

    base_fig = px.bar(resource_df, x='resource', y=['in_use','available','maintenance'],
                      barmode='stack', title="Current Resource Status")
    base_fig.update_layout(height=360, margin=dict(l=8,r=8,t=50,b=8))
    st.plotly_chart(base_fig, use_container_width=True)

    if run_res:
        with st.spinner("Optimizing resources…"):
            res = resource_optimization_basic(resource_df)
            st.session_state["res_results"] = res

    if "res_results" in st.session_state:
        res = st.session_state["res_results"]
        c1, c2 = st.columns(2)
        with c1: st.metric("Solver", res.get("status"))
        with c2: st.metric("Objective (Units Allocated)", f"{res.get('objective_value',0):.1f}" if res.get('objective_value') else "—")

        alloc_df = pd.DataFrame([(k,v) for k,v in res.get("allocation",{}).items()], columns=["Resource","Allocated"])
        fig = px.bar(alloc_df, x="Resource", y="Allocated", title="Optimal Resource Allocation")
        fig.update_layout(height=360, margin=dict(l=8,r=8,t=50,b=8))
        st.plotly_chart(fig, use_container_width=True)

        summary = generate_ai_summary("resource_optimization", res, ai_view_r, resource_df=resource_df)
        if normalize_view(ai_view_r) == "executive":
            st.markdown("### Executive Summary")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")
            st.markdown("**Recommendations:**")
            for r in summary["recs"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("### Technical Deep-Dive")
            st.json(summary["details"])

    with st.expander("📘 Model Details & Implementation"):
        st.markdown("**Utilization LP (demo):** Maximize Σ xᵣ subject to xᵣ ≤ totalᵣ − maintenanceᵣ.")
        st.code(
            """def resource_optimization_basic(resource_df):
    resources = resource_df['resource'].unique()
    total = resource_df.set_index('resource')['total'].to_dict()
    maint = resource_df.set_index('resource')['maintenance'].to_dict()
    prob = LpProblem("Resource_Optimization", LpMaximize)
    x = {r: LpVariable(f"alloc_{r}", lowBound=0, upBound=total[r]) for r in resources}
    prob += lpSum([x[r] for r in resources])
    for r in resources: prob += x[r] <= total[r] - maint[r]
    prob.solve(PULP_CBC_CMD(msg=0))
    return {...}""",
            language="python"
        )

# -----------------------------
# Report & Design Notes
# -----------------------------
with tabs[4]:
    st.markdown("## 🧭 Product & UX Collaboration (Manisha Arora × UX)")
    st.markdown("""
- **Audience modes:** Executive-first summaries with explicit $$ impact; one-tap switch to technical details (constraints, weights).
- **IA:** Tabs map to jobs-to-be-done (Overview → Beds → Staff → Resources). Controls live near charts, no hidden sidebar.
- **Trust:** Always show solver status/objective; reveal assumptions and parameters in expanders with code.
- **Scenarios:** Small set of high-impact controls (demand stress, ICU weight) to explore what-ifs without clutter.
- **Next:** LOS-aware beds, fairness constraints (max consecutive nights), swap-queue for devices, alerts at thresholds.
""")

    # Quick export (beds snapshot if exists)
    if "bed_results" in st.session_state:
        res = st.session_state["bed_results"]
        md = f"# Bed Allocation Report\n\n**Status:** {res.get('status')}\n\n**Objective:** {res.get('objective_value')}\n\n## Allocation\n"
        for d, v in (res.get("allocation",{}) or {}).items():
            md += f"- {d}: {v:.1f}\n"
        st.download_button("⬇️ Download Bed Allocation Snapshot (Markdown)", data=md, file_name="bed_allocation_report.md", type="secondary")
