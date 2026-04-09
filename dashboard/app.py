"""
Dashboard Streamlit — POC LSTM Prédiction de Séries Temporelles
Développé par Stéphane Krebs — Expert IA Freelance
"""

import os
import sys
import threading
import io
import contextlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── Helper Functions ───────────────────────────────────────────────────────

def download_stock(ticker: str = "MC.PA", start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """Download stock data from yfinance or generate synthetic data."""
    from datetime import datetime

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        df = df[["Close"]].copy()
        df.columns = ["close"]
        df.index.name = "date"
        return df
    except:
        # Generate synthetic data fallback
        dates = pd.date_range(start=start, end=end, freq="B")
        n = len(dates)
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.012, n)
        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n) / 252)
        prices = 100 * np.exp(np.cumsum(returns) + seasonal)
        df = pd.DataFrame({"close": prices}, index=dates)
        df.index.name = "date"
        return df

# ─── Config page ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="POC LSTM — Prédiction Boursière",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS dark theme ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Background principal */
    .stApp { background-color: #0e1117; color: #fafafa; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2a3a, #0d1b2a);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
    }
    [data-testid="metric-container"] label { color: #8b949e; font-size: 0.85rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 2rem; font-weight: 700;
    }
    [data-testid="metric-container"] div { color: #ffffff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        color: #8b949e;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #fff !important;
    }

    /* Bouton primaire */
    .stButton > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Footer */
    .footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        text-align: center;
        padding: 8px;
        background-color: #161b22;
        color: #8b949e;
        font-size: 0.78rem;
        border-top: 1px solid #30363d;
        z-index: 100;
    }

    /* Explainer card */
    .explainer {
        background: #161b22;
        border-left: 3px solid #1f6feb;
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 16px;
        color: #c9d1d9;
        font-size: 0.92rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────

TICKERS = {"LVMH (MC.PA)": "MC.PA", "TotalEnergies (TTE.PA)": "TTE.PA"}

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## ⚙️ Paramètres")
    ticker_label = st.selectbox("Titre boursier", list(TICKERS.keys()))
    ticker = TICKERS[ticker_label]

    st.markdown("---")
    start_date = st.date_input("Date de début", value=pd.Timestamp("2018-01-01"))
    end_date   = st.date_input("Date de fin",   value=pd.Timestamp("today"))

    st.markdown("---")
    seq_len = st.slider("Fenêtre glissante (jours)", min_value=10, max_value=120, value=60, step=5)
    epochs  = st.slider("Époques d'entraînement",   min_value=5,  max_value=150, value=50, step=5)

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Modèle : LSTM 2 couches · hidden=64 · dropout=0.2</small>",
        unsafe_allow_html=True,
    )

# ─── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = download_stock(ticker, start=start, end=end)
    return df


def load_model_checkpoint():
    import torch
    path = os.path.join(ROOT, "model", "saved_model.pth")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


@st.cache_resource
def _get_train_function():
    """Import train function une seule fois (cached)."""
    from train import train as train_model
    return train_model


def run_training(ticker, epochs, seq_len):
    """Lance train.train() dans le thread courant et capture stdout."""
    train_model = _get_train_function()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        metrics = train_model(ticker=ticker, epochs=epochs, seq_len=seq_len)
    return metrics, buf.getvalue()


def plotly_dark_layout(**kwargs):
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        **kwargs,
    )

# ─── Chargement données ─────────────────────────────────────────────────────

with st.spinner("Chargement des données…"):
    try:
        df = load_data(ticker, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()

# ─── Onglets ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🏋️ Entraînement", "🔮 Prédictions", "📐 Métriques"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Données
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(f"## Cours historique — {ticker_label}")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Dernier cours", f"{df['close'].iloc[-1]:.2f} €")
    col_b.metric("Min (période)", f"{df['close'].min():.2f} €")
    col_c.metric("Max (période)", f"{df['close'].max():.2f} €")
    col_d.metric("Jours de données", f"{len(df):,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        mode="lines",
        name="Cours de clôture",
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.07)",
    ))
    fig.update_layout(
        **plotly_dark_layout(
            title=f"Prix de clôture — {ticker_label}",
            xaxis_title="Date",
            yaxis_title="Prix (€)",
            height=420,
            hovermode="x unified",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Statistiques descriptives"):
        stats = df["close"].describe().rename({
            "count": "Nb observations", "mean": "Moyenne", "std": "Écart-type",
            "min": "Minimum", "25%": "1er quartile", "50%": "Médiane",
            "75%": "3e quartile", "max": "Maximum",
        })
        st.dataframe(stats.to_frame("Cours de clôture").style.format("{:.2f}"), use_container_width=True)

    with st.expander("Aperçu des données brutes"):
        st.dataframe(df.tail(20).sort_index(ascending=False), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Entraînement
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## Entraînement du modèle LSTM")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.info(f"**Fenêtre :** {seq_len} jours")
    col_info2.info(f"**Époques :** {epochs}")
    col_info3.info(f"**Architecture :** LSTM × 2 → Linear")

    def _plot_loss_curve(ckpt):
        train_l = ckpt["train_losses"]
        val_l   = ckpt["val_losses"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_l, mode="lines", name="Train Loss",
                                  line=dict(color="#58a6ff")))
        fig.add_trace(go.Scatter(y=val_l,   mode="lines", name="Validation Loss",
                                  line=dict(color="#f78166", dash="dash")))
        fig.update_layout(
            **plotly_dark_layout(
                title="Courbe d'apprentissage (MSE)",
                xaxis_title="Époque",
                yaxis_title="MSE Loss",
                height=400,
                hovermode="x unified",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🚀 Lancer l'entraînement"):
        log_placeholder = st.empty()

        with st.spinner("Entraînement en cours… (patientez quelques minutes)"):
            try:
                metrics, logs = run_training(ticker, epochs, seq_len)
                st.success("✅ Entraînement terminé !")

                with log_placeholder.expander("Journal d'entraînement", expanded=False):
                    st.code(logs, language="text")

            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {e}")
                st.exception(e)
                st.stop()

        # Rechargement du checkpoint après entraînement
        ckpt = load_model_checkpoint()
        if ckpt and "train_losses" in ckpt:
            _plot_loss_curve(ckpt)
    else:
        # Affichage courbe si modèle déjà entraîné (et pas en cours d'entraînement)
        ckpt = load_model_checkpoint()
        if ckpt and "train_losses" in ckpt:
            st.markdown(f"_Modèle chargé — entraîné sur **{ckpt.get('ticker', '?')}**_")
            _plot_loss_curve(ckpt)
        elif ckpt is None:
            st.info("Aucun modèle entraîné détecté. Cliquez sur **Lancer l'entraînement** pour commencer.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Prédictions
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## Prédictions vs Réalité")

    pred_path = os.path.join(ROOT, "data", "predictions.csv")

    if not os.path.exists(pred_path):
        st.info("Aucune prédiction disponible. Lancez d'abord l'entraînement dans l'onglet **Entraînement**.")
    else:
        pred_df = pd.read_csv(pred_path, parse_dates=["date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df["date"], y=pred_df["actual"],
            mode="lines", name="Cours réel",
            line=dict(color="#58a6ff", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pred_df["date"], y=pred_df["predicted"],
            mode="lines", name="Prédiction LSTM",
            line=dict(color="#f78166", width=2, dash="dot"),
        ))
        fig.update_layout(
            **plotly_dark_layout(
                title="Comparaison cours réel vs prédiction LSTM (jeu de test)",
                xaxis_title="Date",
                yaxis_title="Prix de clôture (€)",
                height=460,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Résidus
        pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=pred_df["date"], y=pred_df["residual"],
            name="Résidu (réel − prédit)",
            marker_color=np.where(pred_df["residual"] >= 0, "#3fb950", "#f85149"),
        ))
        fig2.update_layout(
            **plotly_dark_layout(
                title="Résidus de prédiction",
                xaxis_title="Date",
                yaxis_title="Écart (€)",
                height=260,
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Données de prédiction (tableau)"):
            st.dataframe(
                pred_df.style.format({"actual": "{:.2f}", "predicted": "{:.2f}", "residual": "{:.2f}"}),
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Métriques
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## Métriques de performance")

    ckpt = load_model_checkpoint()

    if ckpt is None or "metrics" not in ckpt:
        st.info("Aucune métrique disponible. Lancez d'abord l'entraînement.")
    else:
        m = ckpt["metrics"]

        st.markdown("### Résultats sur le jeu de test")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{m['MAE']:.2f} €",
                  help="Erreur Absolue Moyenne — écart moyen entre réel et prédit")
        c2.metric("RMSE", f"{m['RMSE']:.2f} €",
                  help="Racine de l'Erreur Quadratique Moyenne — pénalise les grandes erreurs")
        c3.metric("MAPE", f"{m['MAPE']:.2f} %",
                  help="Erreur Absolue Moyenne en Pourcentage — erreur relative")

        st.markdown("---")

        st.markdown("### Comment interpréter ces résultats ?")
        st.markdown(f"""
<div class="explainer">
<strong>MAE = {m['MAE']:.2f} €</strong><br>
En moyenne, le modèle se trompe de <strong>{m['MAE']:.2f} €</strong> sur le prix de clôture.
C'est l'indicateur le plus intuitif : si le vrai cours est à 700 €, le modèle prédit en moyenne
entre {700 - m['MAE']:.0f} € et {700 + m['MAE']:.0f} €.
<br><br>
<strong>RMSE = {m['RMSE']:.2f} €</strong><br>
Le RMSE pénalise davantage les grosses erreurs. Un RMSE proche du MAE indique que le modèle
fait des erreurs régulières sans gros décrochages ponctuels — signe de stabilité.
<br><br>
<strong>MAPE = {m['MAPE']:.2f} %</strong><br>
En termes relatifs, le modèle se trompe de <strong>{m['MAPE']:.2f} %</strong> en moyenne.
Un MAPE inférieur à 5 % est généralement considéré comme <em>excellent</em> pour une prédiction
financière à horizon court terme. Entre 5 % et 10 % : <em>bon</em>. Au-delà : <em>acceptable</em>.
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Paramètres du modèle sauvegardé")
        hp = ckpt.get("hyperparams", {})
        params_df = pd.DataFrame([
            {"Paramètre": "Ticker", "Valeur": ckpt.get("ticker", "—")},
            {"Paramètre": "Fenêtre (seq_len)", "Valeur": hp.get("seq_len", "—")},
            {"Paramètre": "Couches LSTM", "Valeur": hp.get("num_layers", "—")},
            {"Paramètre": "Taille cachée (hidden)", "Valeur": hp.get("hidden_size", "—")},
            {"Paramètre": "Dropout", "Valeur": hp.get("dropout", "—")},
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">Développé par <strong>Stéphane Krebs</strong> — Expert IA Freelance</div>',
    unsafe_allow_html=True,
)
