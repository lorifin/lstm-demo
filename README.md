# POC LSTM — Prédiction de Séries Temporelles

> **Mission courte, résultat production-ready.**

Démonstration complète d'un modèle LSTM (Long Short-Term Memory) appliqué à la prédiction de cours boursiers, avec dashboard interactif Streamlit. Conçu pour illustrer la valeur d'une approche Deep Learning sur des données temporelles réelles auprès d'une audience DSI/CTO.

---

## Objectif

Valider la capacité d'un réseau LSTM à anticiper les cours de clôture de titres du CAC 40 (LVMH, TotalEnergies) à partir d'une fenêtre glissante de 60 jours de données historiques. Le projet couvre l'intégralité de la chaîne : collecte de données, préparation, entraînement, évaluation et restitution visuelle.

---

## Architecture du projet

```
lstm-demo/
├── data/
│   ├── download_data.py     # Téléchargement yfinance + fallback synthétique
│   ├── raw/                 # Données brutes CSV
│   └── predictions.csv      # Prédictions du dernier run
├── model/
│   ├── lstm_model.py        # Architecture PyTorch
│   └── saved_model.pth      # Checkpoint entraîné
├── dashboard/
│   └── app.py               # Dashboard Streamlit 4 onglets
├── plots/
│   ├── training_loss.png    # Courbe d'apprentissage
│   └── predictions.png      # Graphique réel vs prédit
├── train.py                 # Script d'entraînement CLI
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Cloner le projet
git clone <repo> && cd lstm-demo

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Usage

### Entraînement en ligne de commande

```bash
# Avec les paramètres par défaut (LVMH, 50 époques, fenêtre 60 jours)
python train.py

# Paramètres personnalisés
python train.py --ticker TTE.PA --epochs 100 --seq-len 60

# Aide
python train.py --help
```

### Dashboard interactif

```bash
streamlit run dashboard/app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501) dans le navigateur.

---

## Résultats typiques

| Indicateur | LVMH (MC.PA) | TotalEnergies (TTE.PA) |
|------------|:------------:|:----------------------:|
| MAE        | ~8.50 €      | ~0.90 €                |
| RMSE       | ~11.20 €     | ~1.20 €                |
| MAPE       | ~1.8 %       | ~2.1 %                 |

> Les métriques varient selon la période d'entraînement, le nombre d'époques et les conditions de marché.

---

## Architecture du modèle

```
Input (batch, seq_len=60, 1)
        │
  LSTM Layer 1  (hidden=64, dropout=0.2)
        │
  LSTM Layer 2  (hidden=64)
        │
   Dropout(0.2)
        │
   Linear(64 → 1)
        │
  Output (prix J+1 normalisé)
```

**Optimiseur :** Adam · **Loss :** MSE · **Gradient clipping :** max_norm=1.0
**Split :** 70 % train / 15 % validation / 15 % test
**Normalisation :** MinMaxScaler [0, 1] sur les prix de clôture

---

## Dashboard — Onglets

| Onglet | Contenu |
|--------|---------|
| **Données** | Cours historique interactif + statistiques descriptives |
| **Entraînement** | Bouton de lancement + courbe loss train/validation |
| **Prédictions** | Graphique cours réel vs prédit + graphique des résidus |
| **Métriques** | MAE / RMSE / MAPE en cartes + explication en français clair |

---

## Technologies

- **PyTorch** — réseau LSTM
- **yfinance** — données boursières temps réel
- **Streamlit + Plotly** — dashboard interactif
- **scikit-learn** — normalisation et métriques
- **tqdm** — barre de progression CLI

---

*Développé par **Stéphane Krebs** — Expert IA Freelance*
