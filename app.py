"""
PlantStressDetection — Streamlit Dashboard
Run: streamlit run app.py
"""

import os
import json
import glob
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Stress Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s, border-color .2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: #3fb950; }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #3fb950; margin: 0; }
.metric-label { font-size: .85rem; color: #8b949e; margin: 4px 0 0; text-transform: uppercase; letter-spacing: .08em; }
.metric-delta { font-size: .8rem; margin-top: 4px; }
.delta-up { color: #3fb950; }
.delta-down { color: #f85149; }

/* ── section headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 28px 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #3fb950;
    display: inline-block;
}

/* ── pipeline cards ── */
.pipeline-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.pipeline-title { font-weight: 600; font-size: 1rem; color: #e6edf3; }
.pipeline-desc  { font-size: .82rem; color: #8b949e; margin-top: 4px; }

/* ── status badges ── */
.badge-done    { background:#1a4731; color:#3fb950; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-pending { background:#2d2208; color:#d29922; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-fail    { background:#3d1a1a; color:#f85149; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }

/* ── tabs ── */
[data-testid="stTab"] { font-size:.95rem; font-weight:500; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── buttons ── */
.stButton > button {
    background: #238636;
    color: #fff;
    border: 1px solid #2ea043;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: background .2s;
}
.stButton > button:hover { background: #2ea043; }

/* ── expander ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin-bottom: 32px;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; color: #3fb950; margin: 0; }
.hero p  { font-size: 1.05rem; color: #8b949e; margin: 10px 0 0; }

/* ── log box ── */
.log-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Courier New', monospace;
    font-size: .82rem;
    color: #3fb950;
    max-height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")
DATA_DIR    = Path("data/raw")

def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def fmt(val, decimals=4):
    if val == "N/A" or val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)

def has_results(subpath: str) -> bool:
    return (RESULTS_DIR / subpath).exists()

def result_badge(subpath: str) -> str:
    if has_results(subpath):
        return '<span class="badge-done">✓ Done</span>'
    return '<span class="badge-pending">○ Pending</span>'

def dark_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        title_font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#30363d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#30363d", zerolinecolor="#30363d")
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 PlantStress")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔬 Pipelines", "📊 Results", "🧪 Studies", "🚀 Run Experiments", "📄 Research Paper"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick status
    st.markdown("**Results Status**")
    checks = {
        "P1 · Texture SVM":     "pipeline1_metrics.json",
        "P2 · MobileNetV3":     "mobilenet/test_metrics.json",
        "P3 · Severity Reg.":   "regression/test_metrics.json",
        "P4 · Anomaly Det.":    "anomaly/test_metrics.json",
        "P5 · Entropy Drift":   "entropy_drift/drift_metrics.json",
        "P6 · AlexNet":         "alexnet/test_metrics.json",
        "Cross-Crop Study":     "studies/cross_crop_metrics.json",
        "Robustness Study":     "studies/robustness_metrics.json",
        "Deployment Study":     "studies/deployment_metrics.json",
    }
    for label, path in checks.items():
        exists = (RESULTS_DIR / path).exists()
        icon = "🟢" if exists else "⚪"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.markdown(f"<small style='color:#8b949e'>Last refresh: {datetime.now().strftime('%H:%M:%S')}</small>", unsafe_allow_html=True)
    if st.button("🔄 Refresh"):
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
        <h1>🌿 Plant Stress Detection</h1>
        <p>Early detection of plant disease via Texture Entropy Drift · Deep Feature Embedding · MobileNetV3</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level metrics ──
    p1 = load_json(RESULTS_DIR / "pipeline1_metrics.json")
    p2 = load_json(RESULTS_DIR / "mobilenet/test_metrics.json")
    p3 = load_json(RESULTS_DIR / "regression/test_metrics.json")
    p4 = load_json(RESULTS_DIR / "anomaly/test_metrics.json")
    dep = load_json(RESULTS_DIR / "studies/deployment_metrics.json")

    c1, c2, c3, c4, c5 = st.columns(5)
    def metric_html(value, label, delta=None, good="high"):
        delta_html = ""
        if delta:
            cls = "delta-up" if good == "high" else "delta-down"
            delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
        return f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>"""

    svm_acc  = fmt(p1.get("accuracy"), 3) if p1 else "—"
    mob_acc  = fmt(p2.get("accuracy"), 3) if p2 else "—"
    reg_r2   = fmt(p3.get("r2"), 3) if p3 else "—"
    ano_auc  = fmt(p4.get("Isolation_Forest_AUC"), 3) if p4 else "—"
    lat_ms   = f"{dep.get('MobileNet',{}).get('Latency_ms','—')} ms" if dep else "—"

    c1.markdown(metric_html(svm_acc,  "SVM Accuracy"),      unsafe_allow_html=True)
    c2.markdown(metric_html(mob_acc,  "MobileNet Accuracy"), unsafe_allow_html=True)
    c3.markdown(metric_html(reg_r2,   "Severity R²"),        unsafe_allow_html=True)
    c4.markdown(metric_html(ano_auc,  "Anomaly AUC"),        unsafe_allow_html=True)
    c5.markdown(metric_html(lat_ms,   "Edge Latency", good="low"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture overview ──
    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-header">Framework Architecture</div>', unsafe_allow_html=True)

        pipelines = [
            ("P1", "Classical Texture + SVM",         "GLCM · LBP · Color Histogram → RBF-SVM Classifier",         "pipeline1_metrics.json"),
            ("P2", "MobileNetV3 Classification",      "Two-phase fine-tuning · Discriminative LRs · 30-epoch train","mobilenet/test_metrics.json"),
            ("P6", "AlexNet Classification",          "Custom PyTorch implementation matching Keras architecture",  "alexnet/test_metrics.json"),
            ("P3", "Severity Regression",             "Lesion proxy score · Sigmoid head · MSE optimization",       "regression/test_metrics.json"),
            ("P4", "Deep Embedding Anomaly",          "GAP embeddings · One-Class SVM · Isolation Forest · AE",     "anomaly/test_metrics.json"),
            ("P5", "Texture Entropy Drift",           "GLCM entropy baseline · Mahalanobis distance drift metric",  "entropy_drift/drift_metrics.json"),
        ]
        for tag, title, desc, path in pipelines:
            badge = result_badge(path)
            st.markdown(f"""
            <div class="pipeline-card">
                <div class="pipeline-title">{tag} · {title} &nbsp; {badge}</div>
                <div class="pipeline-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Dataset Distribution</div>', unsafe_allow_html=True)

        if DATA_DIR.exists():
            classes = [d for d in DATA_DIR.iterdir() if d.is_dir()]
            if classes:
                labels, counts = [], []
                for c in sorted(classes):
                    imgs = list(c.glob("*.jpg")) + list(c.glob("*.JPG"))
                    labels.append(c.name.replace("___", "\n").replace(",_bell", ""))
                    counts.append(len(imgs))

                fig = px.bar(
                    x=counts, y=labels, orientation="h",
                    color=counts,
                    color_continuous_scale=["#1a4731","#2ea043","#3fb950"],
                    labels={"x": "Images", "y": ""},
                )
                fig.update_layout(
                    coloraxis_showscale=False,
                    height=340,
                    yaxis=dict(tickfont=dict(size=10)),
                )
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dataset found. Go to **Run Experiments** and generate dummy data first.")
        else:
            st.info("Dataset not yet generated. Use the **Run Experiments** page.")

    # ── Comparison bar ──
    if p1 and p2:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        metrics_list = ["accuracy", "precision", "recall", "f1"]
        svm_vals = [p1.get(m, 0) for m in metrics_list]
        mob_vals = [p2.get(m, 0) for m in metrics_list]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Texture + SVM", x=[m.capitalize() for m in metrics_list], y=svm_vals,
                              marker_color="#388bfd", text=[f"{v:.3f}" for v in svm_vals], textposition="outside"))
        fig2.add_trace(go.Bar(name="MobileNetV3",   x=[m.capitalize() for m in metrics_list], y=mob_vals,
                              marker_color="#3fb950", text=[f"{v:.3f}" for v in mob_vals], textposition="outside"))
        fig2.update_layout(
            barmode="group", title="Classification Metrics: SVM vs MobileNetV3",
            yaxis=dict(range=[0, 1.15], title="Score"),
            height=360,
        )
        dark_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Pipelines
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔬 Pipelines":
    st.markdown("## 🔬 Pipeline Deep-Dives")

    tab1, tab2, tab6, tab3, tab4, tab5 = st.tabs([
        "P1 · Texture SVM", "P2 · MobileNetV3", "P6 · AlexNet", "P3 · Severity Reg.",
        "P4 · Anomaly Det.", "P5 · Entropy Drift"
    ])

    # ── P1 ──
    with tab1:
        st.markdown("### Classical Texture Features + SVM")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            **Feature Extraction:**
            - GLCM (contrast, correlation, energy, homogeneity) at distances [1, 3, 5] and 4 angles
            - LBP (radius=3, 24 points, uniform method)
            - HSV colour histograms (8 bins × 3 channels)

            **Classifier:** RBF-kernel SVM with StandardScaler pipeline

            **Strengths:** Interpretable, fast inference, no GPU required
            **Weaknesses:** Manual feature engineering, sensitive to lighting
            """)
        with col2:
            p1 = load_json(RESULTS_DIR / "pipeline1_metrics.json")
            if p1:
                metrics = {k.capitalize(): v for k, v in p1.items() if isinstance(v, float)}
                fig = go.Figure(go.Bar(
                    x=list(metrics.keys()), y=list(metrics.values()),
                    marker=dict(color=["#388bfd","#3fb950","#d29922","#f85149"]),
                    text=[f"{v:.4f}" for v in metrics.values()],
                    textposition="outside",
                ))
                fig.update_layout(title="Pipeline 1 · Test Metrics", yaxis=dict(range=[0, 1.15]), height=320)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results yet. Run Pipeline 1 first.")

    # ── P2 ──
    with tab2:
        st.markdown("### MobileNetV3-Large Fine-Tuning")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            **Architecture:** MobileNetV3-Large (ImageNet pre-trained)
            **Strategy:** Two-phase training
            1. Freeze backbone → train head (15 epochs)
            2. Unfreeze top 30% → discriminative LRs (15 epochs)

            **Optimiser:** Adam | **LR:** head=1e-3, backbone=1e-4
            **Loss:** CrossEntropyLoss
            """)
        with col2:
            p2 = load_json(RESULTS_DIR / "mobilenet/test_metrics.json")
            if p2:
                fig = go.Figure(go.Bar(
                    x=[k.capitalize() for k in p2 if isinstance(p2[k], float)],
                    y=[v for v in p2.values() if isinstance(v, float)],
                    marker_color="#3fb950",
                    text=[f"{v:.4f}" for v in p2.values() if isinstance(v, float)],
                    textposition="outside",
                ))
                fig.update_layout(title="Pipeline 2 · Test Metrics", yaxis=dict(range=[0, 1.15]), height=320)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results yet. Run Pipeline 2 first.")

        # Training history image
        hist_img = RESULTS_DIR / "mobilenet/training_history.png"
        cm_img   = RESULTS_DIR / "mobilenet/confusion_matrix.png"
        roc_img  = RESULTS_DIR / "mobilenet/roc_curve.png"

        img_cols = [p for p in [hist_img, cm_img, roc_img] if p.exists()]
        if img_cols:
            cols = st.columns(len(img_cols))
            titles = ["Training History", "Confusion Matrix", "ROC Curve"]
            for i, (col, imgp) in enumerate(zip(cols, img_cols)):
                col.image(str(imgp), caption=titles[i], use_container_width=True)

    # ── P3 ──
    with tab3:
        st.markdown("### Continuous Severity Regression")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            **Target:** Lesion-to-leaf area ratio (colour-segmentation proxy)
            **Model:** MobileNetV3-Small with Sigmoid regression head
            **Loss:** MSE

            **Severity Stages:**
            - 🟢 Mild   — < 5% lesion coverage
            - 🟡 Moderate — 5–20% coverage
            - 🔴 Severe — > 20% coverage
            """)
        with col2:
            p3 = load_json(RESULTS_DIR / "regression/test_metrics.json")
            if p3:
                fig = go.Figure()
                for key, val, color in [("MAE", p3.get("mae", 0), "#f85149"), ("R²", p3.get("r2", 0), "#3fb950")]:
                    fig.add_trace(go.Bar(name=key, x=[key], y=[val], marker_color=color,
                                         text=[f"{val:.4f}"], textposition="outside"))
                fig.update_layout(title="Pipeline 3 · Regression Metrics", height=320, showlegend=False)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results yet. Run Pipeline 3 first.")

        err_img = RESULTS_DIR / "regression/error_distribution.png"
        if err_img.exists():
            st.image(str(err_img), caption="Error Distribution & Scatter", use_container_width=True)

    # ── P4 ──
    with tab4:
        st.markdown("### Deep Embedding Anomaly Detection")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            **Embeddings:** 960-dim GAP from MobileNetV3-Large backbone
            **Models Evaluated:**
            - One-Class SVM (nu=0.1)
            - Isolation Forest (contamination=0.1)
            - Autoencoder (960→512→128→32→512→960)

            **Evaluation:** ROC-AUC (Healthy vs. Diseased)
            """)
        with col2:
            p4 = load_json(RESULTS_DIR / "anomaly/test_metrics.json")
            if p4:
                models_names = ["One-Class SVM", "Isolation Forest", "Autoencoder"]
                keys = ["OCSVM_AUC", "Isolation_Forest_AUC", "Autoencoder_AUC"]
                aucs = [p4.get(k, 0) for k in keys]
                fig = go.Figure(go.Bar(
                    x=models_names, y=aucs,
                    marker=dict(color=["#388bfd","#3fb950","#d29922"]),
                    text=[f"{v:.4f}" for v in aucs], textposition="outside",
                ))
                fig.update_layout(title="Pipeline 4 · Anomaly AUC Scores", yaxis=dict(range=[0, 1.15]), height=320)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results yet. Run Pipeline 4 first.")

        roc_img = RESULTS_DIR / "anomaly/anomaly_roc.png"
        emb_img = RESULTS_DIR / "anomaly/embedding_pca.png"
        for img, cap in [(roc_img, "Anomaly ROC Curves"), (emb_img, "Embedding PCA Visualisation")]:
            if img.exists():
                st.image(str(img), caption=cap, use_container_width=True)

    # ── P5 ──
    with tab5:
        st.markdown("### Texture Entropy Drift (Mahalanobis)")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            **Method:**
            1. Extract GLCM entropy for all healthy training images
            2. Fit multivariate normal distribution as baseline
            3. Compute Mahalanobis distance for test images
            4. Higher distance → more severe stress

            **Hypothesis:** Drift in texture entropy is an early-stage biomarker
            """)
        with col2:
            p5 = load_json(RESULTS_DIR / "entropy_drift/drift_metrics.json")
            if p5:
                stages = list(p5.keys())
                means  = [p5[s].get("mean_dist", 0) for s in stages]
                stds   = [p5[s].get("std_dist",  0) for s in stages]
                colors = {"Healthy": "#3fb950", "Mild": "#d29922", "Severe": "#f85149"}
                fig = go.Figure()
                for s, m, e in zip(stages, means, stds):
                    fig.add_trace(go.Bar(
                        name=s, x=[s], y=[m],
                        error_y=dict(type="data", array=[e], visible=True),
                        marker_color=colors.get(s, "#8b949e"),
                        text=[f"{m:.3f}"], textposition="outside",
                    ))
                fig.update_layout(title="Mahalanobis Distance by Severity", showlegend=False, height=320)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results yet. Run Pipeline 5 first.")

        drift_img = RESULTS_DIR / "entropy_drift/drift_curves.png"
        if drift_img.exists():
            st.image(str(drift_img), caption="Entropy Drift Curves", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Results":
    st.markdown("## 📊 Results Dashboard")

    # ── Summary table ──
    st.markdown('<div class="section-header">All Metrics at a Glance</div>', unsafe_allow_html=True)

    p1  = load_json(RESULTS_DIR / "pipeline1_metrics.json")
    p2  = load_json(RESULTS_DIR / "mobilenet/test_metrics.json")
    p6  = load_json(RESULTS_DIR / "alexnet/test_metrics.json")
    p3  = load_json(RESULTS_DIR / "regression/test_metrics.json")
    p4  = load_json(RESULTS_DIR / "anomaly/test_metrics.json")
    p5  = load_json(RESULTS_DIR / "entropy_drift/drift_metrics.json")

    rows = []
    if p1: rows.append({
        "Pipeline": "P1 · Texture SVM", "Accuracy": fmt(p1.get("accuracy")),
        "Precision": fmt(p1.get("precision")), "Recall": fmt(p1.get("recall")),
        "F1": fmt(p1.get("f1")), "Extra": "—"
    })
    if p2: rows.append({
        "Pipeline": "P2 · MobileNetV3", "Accuracy": fmt(p2.get("accuracy")),
        "Precision": fmt(p2.get("precision")), "Recall": fmt(p2.get("recall")),
        "F1": fmt(p2.get("f1")), "Extra": "—"
    })
    if p3: rows.append({
        "Pipeline": "P3 · Severity Reg.", "Accuracy": "—",
        "Precision": "—", "Recall": "—", "F1": "—",
        "Extra": f"MAE={fmt(p3.get('mae'))} R²={fmt(p3.get('r2'))}"
    })
    if p4: rows.append({
        "Pipeline": "P4 · Anomaly Det.", "Accuracy": "—",
        "Precision": "—", "Recall": "—", "F1": "—",
        "Extra": f"IsoForest AUC={fmt(p4.get('Isolation_Forest_AUC'))}"
    })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No results found. Run the experiments first.")

    # ── Radar chart ──
    if p1 and p2:
        st.markdown('<div class="section-header">Performance Radar</div>', unsafe_allow_html=True)
        cats = ["Accuracy", "Precision", "Recall", "F1"]
        svm_vals = [p1.get(m.lower(), 0) for m in cats]
        mob_vals = [p2.get(m.lower(), 0) for m in cats]
        svm_vals += [svm_vals[0]]
        mob_vals += [mob_vals[0]]
        cats_closed = cats + [cats[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=svm_vals, theta=cats_closed, fill="toself",
                                       name="Texture SVM", line_color="#388bfd"))
        fig.add_trace(go.Scatterpolar(r=mob_vals, theta=cats_closed, fill="toself",
                                       name="MobileNetV3", line_color="#3fb950"))
        fig.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(range=[0, 1], gridcolor="#30363d", tickfont_color="#8b949e"),
                angularaxis=dict(gridcolor="#30363d"),
            ),
            title="Model Performance Radar",
            height=420,
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly + Drift side by side ──
    if p4 or p5:
        col_l, col_r = st.columns(2)

        with col_l:
            if p4:
                st.markdown('<div class="section-header">Anomaly AUC Comparison</div>', unsafe_allow_html=True)
                labels_a = ["OCSVM", "Isolation Forest", "Autoencoder"]
                keys_a   = ["OCSVM_AUC", "Isolation_Forest_AUC", "Autoencoder_AUC"]
                vals_a   = [p4.get(k, 0) for k in keys_a]
                fig = go.Figure(go.Bar(
                    x=labels_a, y=vals_a,
                    marker=dict(color=["#388bfd","#3fb950","#d29922"]),
                    text=[f"{v:.3f}" for v in vals_a], textposition="outside"
                ))
                fig.update_layout(yaxis=dict(range=[0, 1.15]), height=320)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if p5:
                st.markdown('<div class="section-header">Entropy Drift by Stage</div>', unsafe_allow_html=True)
                stages  = list(p5.keys())
                means_v = [p5[s].get("mean_dist", 0) for s in stages]
                colors  = {"Healthy": "#3fb950", "Mild": "#d29922", "Severe": "#f85149"}
                fig = go.Figure(go.Bar(
                    x=stages, y=means_v,
                    marker_color=[colors.get(s, "#8b949e") for s in stages],
                    text=[f"{v:.3f}" for v in means_v], textposition="outside"
                ))
                fig.update_layout(height=320)
                dark_fig(fig)
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Studies
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🧪 Studies":
    st.markdown("## 🧪 Additional Studies")

    tab_cc, tab_rb, tab_dp = st.tabs(["Cross-Crop Generalisation", "Robustness Study", "Deployment Analysis"])

    # ── Cross-Crop ──
    with tab_cc:
        st.markdown("### Cross-Crop Generalisation Study")
        st.markdown("""
        **Setup:** Train on Tomato + Potato · Test on Pepper + Apple
        **Goal:** Measure how well the model generalises to unseen crop species
        """)
        cross = load_json(RESULTS_DIR / "studies/cross_crop_metrics.json")
        if cross:
            c1, c2, c3 = st.columns(3)
            c1.metric("Seen Crops Accuracy",   fmt(cross.get("Seen_Crops_Accuracy")))
            c2.metric("Unseen Crops Accuracy",  fmt(cross.get("Unseen_Crops_Accuracy")))
            c3.metric("Generalisation Drop",    fmt(cross.get("Generalization_Degradation")))

            seen   = cross.get("Seen_Crops_Accuracy", 0)
            unseen = cross.get("Unseen_Crops_Accuracy", 0)
            fig = go.Figure(go.Bar(
                x=["Seen Crops (Train Species)", "Unseen Crops (New Species)"],
                y=[seen, unseen],
                marker_color=["#3fb950", "#f85149"],
                text=[f"{v:.4f}" for v in [seen, unseen]],
                textposition="outside",
            ))
            fig.update_layout(title="Cross-Crop Accuracy Drop", yaxis=dict(range=[0, 1.15]), height=360)
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cross-crop results found. Run the cross-crop study first.")

    # ── Robustness ──
    with tab_rb:
        st.markdown("### Robustness to Environmental Perturbations")
        st.markdown("""
        **Perturbations applied to test set:**
        - **Baseline** — clean test images
        - **Blur** — Gaussian blur (kernel 7–11)
        - **Noise** — Gaussian noise (var 100–200)
        - **Low Brightness** — brightness −40%
        """)
        robust = load_json(RESULTS_DIR / "studies/robustness_metrics.json")
        if robust:
            conditions = list(robust.keys())
            accs = [robust[c] for c in conditions]
            colors_r = ["#3fb950" if c == "baseline" else "#d29922" if robust[c] > 0.5 else "#f85149"
                        for c in conditions]
            fig = go.Figure(go.Bar(
                x=[c.replace("_", " ").title() for c in conditions],
                y=accs,
                marker_color=colors_r,
                text=[f"{v:.4f}" for v in accs],
                textposition="outside",
            ))
            fig.add_hline(y=robust.get("baseline", 0), line_dash="dash", line_color="#388bfd",
                          annotation_text="Baseline", annotation_position="top right")
            fig.update_layout(title="Model Accuracy Under Perturbations",
                              yaxis=dict(range=[0, 1.15]), height=380)
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            baseline_acc = robust.get("baseline", 1)
            st.markdown("**Degradation from Baseline:**")
            for cond in conditions:
                if cond != "baseline":
                    drop = (baseline_acc - robust[cond]) / baseline_acc * 100
                    color = "🟢" if drop < 5 else "🟡" if drop < 20 else "🔴"
                    st.markdown(f"{color} **{cond.replace('_',' ').title()}:** {drop:.1f}% drop")
        else:
            st.info("No robustness results found. Run the robustness study first.")

    # ── Deployment ──
    with tab_dp:
        st.markdown("### Edge Deployment Feasibility")
        st.markdown("""
        **Goal:** Evaluate viability of deploying models on agricultural edge devices (e.g. Raspberry Pi)
        **Measured on:** CPU (simulated edge device)
        """)
        deploy = load_json(RESULTS_DIR / "studies/deployment_metrics.json")
        if deploy:
            rows_d = []
            for model_name, stats in deploy.items():
                rows_d.append({
                    "Model":        model_name.replace("_", " "),
                    "Size (MB)":    stats.get("Size_MB", "—"),
                    "Latency (ms)": stats.get("Latency_ms", stats.get("Latency_w_Features_ms", "—")),
                    "Latency Std":  stats.get("Latency_std_ms", "—"),
                })
            df_d = pd.DataFrame(rows_d)
            st.dataframe(df_d, use_container_width=True, hide_index=True)

            # Latency comparison chart
            model_names = [r["Model"] for r in rows_d]
            latencies   = [float(r["Latency (ms)"]) if r["Latency (ms)"] != "—" else 0 for r in rows_d]
            sizes       = [float(r["Size (MB)"]) if r["Size (MB)"] != "—" else 0 for r in rows_d]

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Inference Latency (ms)", "Model Size (MB)"])
            fig.add_trace(go.Bar(x=model_names, y=latencies, marker_color=["#388bfd","#3fb950"],
                                  text=[f"{v:.1f}" for v in latencies], textposition="outside"), row=1, col=1)
            fig.add_trace(go.Bar(x=model_names, y=sizes, marker_color=["#d29922","#f85149"],
                                  text=[f"{v:.1f}" for v in sizes], textposition="outside"), row=1, col=2)
            fig.update_layout(showlegend=False, height=360)
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No deployment results found. Run the deployment study first.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Run Experiments
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🚀 Run Experiments":
    st.markdown("## 🚀 Run Experiments")
    st.markdown("Select which steps to execute, then click **Run**. Output streams below in real time.")

    col_cfg, col_run = st.columns([1, 1.8])

    with col_cfg:
        st.markdown("**Configuration**")
        gen_data  = st.checkbox("Generate dummy dataset", value=True)
        run_p1    = st.checkbox("P1 · Texture SVM",                value=True)
        run_p2    = st.checkbox("P2 · MobileNetV3 Classification",  value=True)
        run_p6    = st.checkbox("P6 · AlexNet Classification",      value=True)
        run_p3    = st.checkbox("P3 · Severity Regression",         value=True)
        run_p4    = st.checkbox("P4 · Anomaly Detection",           value=True)
        run_p5    = st.checkbox("P5 · Entropy Drift",               value=True)
        run_cross = st.checkbox("Study · Cross-Crop",               value=True)
        run_rob   = st.checkbox("Study · Robustness",               value=True)
        run_dep   = st.checkbox("Study · Deployment",               value=True)
        run_paper = st.checkbox("Generate Research Paper",          value=True)

        st.markdown("---")
        dry_run = st.toggle("Dry-run mode (dummy data)", value=True)

    with col_run:
        st.markdown("**Execution Log**")
        log_area = st.empty()

        if st.button("▶  Run Selected Experiments", use_container_width=True):
            import subprocess, sys as _sys

            args_list = [_sys.executable, "run_experiments.py"]
            if dry_run:
                args_list.append("--dry-run")
            if not gen_data:
                args_list.append("--skip-data")

            only_flags = []
            if run_p1:    only_flags.append("p1")
            if run_p2:    only_flags.append("p2")
            if run_p6:    only_flags.append("p6")
            if run_p3:    only_flags.append("p3")
            if run_p4:    only_flags.append("p4")
            if run_p5:    only_flags.append("p5")
            if run_cross: only_flags.append("cross")
            if run_rob:   only_flags.append("robust")
            if run_dep:   only_flags.append("deploy")
            if run_paper: only_flags.append("paper")

            if only_flags:
                args_list += ["--only"] + only_flags

            log_lines = []

            with st.spinner("Running experiments…"):
                proc = subprocess.Popen(
                    args_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(Path(__file__).parent),
                )
                for line in proc.stdout:
                    log_lines.append(line.rstrip())
                    log_area.markdown(
                        '<div class="log-box">' + "\n".join(log_lines[-60:]) + "</div>",
                        unsafe_allow_html=True,
                    )
                proc.wait()

            if proc.returncode == 0:
                st.success("✅ All selected experiments completed! Navigate to **Results** to view outputs.")
            else:
                st.error("❌ Some experiments failed. See log above for details.")

            st.rerun()

    # ── Quick commands reference ──
    st.markdown("---")
    st.markdown("### Quick Terminal Commands")
    st.code("""# Generate dummy data + run all pipelines
python run_experiments.py --dry-run

# Run only specific pipelines
python run_experiments.py --dry-run --only p1 p2 p5

# Skip data generation (if data already exists)
python run_experiments.py --skip-data

# Launch this dashboard
streamlit run app.py
""", language="bash")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Research Paper
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📄 Research Paper":
    st.markdown("## 📄 Auto-Generated Research Paper")

    paper_path = RESULTS_DIR / "final_research_paper.md"

    col_gen, col_dl = st.columns([1, 3])
    with col_gen:
        if st.button("🔄 Regenerate Paper"):
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent))
                from tools.generate_paper import generate_paper
                generate_paper()
                st.success("Paper regenerated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if paper_path.exists():
        content = paper_path.read_text()

        with col_dl:
            st.download_button(
                label="⬇ Download Markdown",
                data=content,
                file_name="PlantStressDetection_Paper.md",
                mime="text/markdown",
                use_container_width=False,
            )

        st.markdown("---")
        st.markdown(content)
    else:
        st.info(
            "Paper not yet generated. Run all experiments first, then click **Regenerate Paper** above "
            "or run `python run_experiments.py --dry-run --only paper` from the terminal."
        )
