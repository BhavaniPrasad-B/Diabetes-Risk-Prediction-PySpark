import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Pima Clinical Insights",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed"
)

# 2. Refined Aesthetic CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Tenor+Sans&display=swap');

.stApp { background: #1e2124; color: #e2e8f0; }
html, body, [class*="css"] { font-family: 'Tenor Sans', sans-serif; }
h1, h2, h3, .royal-header { 
    font-family: 'Playfair Display', serif !important; 
    font-weight: 700; 
    color: #f8fafc; 
    margin-top: 2rem !important;
}

.hero {
    background: #16181b;
    border-radius: 20px;
    padding: 4rem;
    margin-bottom: 3rem;
    border: 1px solid #2d3135;
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
}
.hero-tag { font-family: 'Tenor Sans'; font-size: 0.75rem; text-transform: uppercase; color: #a18a68; letter-spacing: 3px; }
.hero-title { font-size: 4rem; color: #ffffff; line-height: 1; margin: 15px 0; letter-spacing: -0.02em; }
.hero-title em { font-family: 'Playfair Display', serif; font-style: italic; color: #a18a68; }

.scard {
    background: #25282c;
    border-radius: 16px;
    padding: 2.5rem;
    border-top: 4px solid #7f1d1d;
    border-bottom: 1px solid #2d3135;
    border-left: 1px solid #2d3135;
    border-right: 1px solid #2d3135;
    text-align: center;
    transition: transform 0.3s ease;
}
.scard:hover { transform: translateY(-5px); border-color: #7f1d1d; }
.scard-label { font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 2px; font-weight: 600; }
.scard-value { font-size: 3.5rem; font-weight: 700; color: #f8fafc; font-family: 'Playfair Display', serif; margin: 10px 0; }

.note-panel { background: #16181b; border-radius: 20px; padding: 3rem; border: 1px solid #2d3135; margin-top: 2.5rem; }
.chart-card { background: #25282c; border-radius: 20px; padding: 3rem; border: 1px solid #2d3135; }
</style>
""", unsafe_allow_html=True)

try:
    metrics_df = pd.read_csv("metrics.csv")
    importance_df = pd.read_csv("importance.csv").sort_values("Score", ascending=True)
    raw_df = pd.read_csv("diabetes.csv")

    f1, prec, rec = metrics_df['F1-Score'][0], metrics_df['Precision'][0], metrics_df['Recall'][0]

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">Clinical Analytics / Project </div>
        <div class="hero-title">Diabetes <em>Risk</em><br>Insight Engine</div>
        <p style="color:#94a3b8; font-size:1.15rem; margin-top:20px; font-weight:300; max-width: 800px;">
            A high-fidelity analysis of Pima health parameters using Distributed Spark Computing and Ensemble Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 class='royal-header'>Model Performance Metrics</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            f"""<div class="scard"><div class="scard-label">F1-Score</div><div class="scard-value">{f1:.4f}</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"""<div class="scard"><div class="scard-label">Precision</div><div class="scard-value" style="color:#cbd5e1;">{prec:.4f}</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"""<div class="scard"><div class="scard-label">Recall</div><div class="scard-value" style="color:#cbd5e1;">{rec:.4f}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<h3 class='royal-header'>Clinical Driver Significance</h3>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=importance_df['Score'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(color=importance_df['Score'], colorscale=[[0, '#2d3135'], [0.4, '#a18a68'], [1, '#7f1d1d']]),
        text=[f" {s:.3f}" for s in importance_df['Score']],
        textposition='outside',
        textfont=dict(family='Tenor Sans', color='#f8fafc', size=14)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'), margin=dict(l=0, r=120, t=20, b=0),
        height=550,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title="Importance Weight"),
        yaxis=dict(showgrid=False, tickfont=dict(size=16))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="note-panel">
        <h3 class='royal-header' style='margin-top:0;'>Pipeline Strategy & Interpretation</h3>
        <div style="display:flex; gap:80px; margin-top:30px;">
            <div style="flex:1; background: rgba(255,255,255,0.02); padding: 25px; border-radius: 12px;">
                <p style="color:#a18a68; font-weight:700; text-transform:uppercase; letter-spacing:2px; font-size:0.85rem;">Clinical Context</p>
                <p style="font-size:1.1rem; line-height:1.8; color:#cbd5e1; font-weight:300;">
                    The Random Forest ensemble identifies <b>{importance_df.iloc[-1]['Feature']}</b> as the primary driver of risk. This validates physiological expectations in metabolic diagnostics.
                </p>
            </div>
            <div style="flex:1; background: rgba(255,255,255,0.02); padding: 25px; border-radius: 12px; border-left: 2px solid #7f1d1d;">
                <p style="color:#a18a68; font-weight:700; text-transform:uppercase; letter-spacing:2px; font-size:0.85rem;">Engineering Protocol</p>
                <ul style="font-size:1rem; line-height:2.1; color:#94a3b8; margin:0; padding-left:20px; font-weight:300;">
                    <li>Missing clinical measurements imputed via Mean calculation.</li>
                    <li>Z-Score scaling applied to ensure parameter magnitude parity.</li>
                    <li>Distributed PySpark MLlib Ensemble Classification.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Engine Data Missing: Run 'python3 process.py' first.")