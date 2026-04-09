# 🚀 PROTOCOLE DE LANCEMENT - LSTM Dashboard Streamlit

## 📋 Prérequis

- Python 3.11.5 (via PyEnv)
- Dépendances installées: `pip install -r requirements.txt`
- Modèle LSTM entraîné dans `models/checkpoints/`
- API LSTM running sur port 8000 (optionnel mais recommandé)

---

## ✅ CHECKLIST PRÉ-LANCEMENT

### 1️⃣ Vérifier les dépendances

```bash
cd /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo

# Vérifier Python version
python --version
# → Doit afficher: Python 3.11.5

# Vérifier les packages critiques
python -c "import streamlit; import torch; import sklearn; import yfinance; print('✓ All packages OK')"
```

### 2️⃣ Vérifier le modèle

```bash
# Le modèle doit exister
ls -lah models/checkpoints/
# → Doit contenir: latest.pt, shadow_model.pt, final_model.pt

# Vérifier le checkpoint est lisible
python -c "import torch; ckpt = torch.load('models/checkpoints/latest.pt', map_location='cpu'); print(f'✓ Model checkpoint OK: {len(ckpt)} keys')"
```

### 3️⃣ Vérifier les données

```bash
# Le dossier data doit exister
ls -la data/
# → Doit contenir ou créer: stock_data.csv, predictions.csv
```

---

## 🚀 LANCEMENT STREAMLIT

### **Option A: Lancement Simple (Recommandé)**

```bash
cd /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo/dashboard

# Démarrer l'app
streamlit run app.py
```

**Sortie attendue:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Folder: /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo/dashboard
```

Ouvre automatiquement dans le navigateur. ✅

---

### **Option B: Lancement avec Configuration Personnalisée**

```bash
cd /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo/dashboard

# Lancer sur port personnalisé
streamlit run app.py --server.port 8502

# Lancer en mode headless (sans navigateur auto-ouvert)
streamlit run app.py --server.headless true

# Lancer en mode debug
streamlit run app.py --logger.level=debug
```

---

## 🔧 TROUBLESHOOTING

### **Problème: "ModuleNotFoundError: No module named 'yfinance'"**

```bash
# Solution
pip install yfinance pandas numpy torch scikit-learn streamlit plotly
```

### **Problème: "deadlock detected by _ModuleLock"**

**C'est RÉSOLU** dans `app.py` avec:
- `@st.cache_resource` sur l'import dynamique
- Import sklearn en haut du fichier

**Si ça arrive quand même:**
```bash
# Clear le cache Streamlit
streamlit cache clear

# Relancer
streamlit run app.py
```

### **Problème: "Port 8501 already in use"**

```bash
# Tuer le process existant
lsof -i :8501 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Ou utiliser un autre port
streamlit run app.py --server.port 8502
```

### **Problème: "Model not loading"**

```bash
# Vérifier le checkpoint
python -c "
import torch
try:
    ckpt = torch.load('models/checkpoints/latest.pt', map_location='cpu', weights_only=False)
    print(f'✓ Checkpoint loaded: {list(ckpt.keys())}')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

---

## 📊 STRUCTURE REQUISE

```
lstm-demo/
├── dashboard/
│   └── app.py                 ← Fichier principal Streamlit
├── models/
│   └── checkpoints/
│       ├── latest.pt          ← Modèle production
│       ├── shadow_model.pt    ← Modèle shadow
│       └── final_model.pt     ← Modèle final
├── data/
│   ├── stock_data.csv         ← Données historiques
│   └── predictions.csv        ← Prédictions
├── train.py                   ← Script d'entraînement
├── requirements.txt           ← Dépendances
└── README.md
```

---

## 🎯 CHECKLIST DE DÉMARRAGE RAPIDE

**Copie-colle cette commande pour démarrer:**

```bash
cd /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo/dashboard && \
echo "✓ Vérification dépendances..." && \
python -c "import streamlit; import torch; import sklearn; print('✓ OK')" && \
echo "✓ Dossier models..." && \
ls -q models/checkpoints/latest.pt >/dev/null && echo "✓ Modèle trouvé" || echo "✗ Modèle manquant" && \
echo "🚀 Lancement Streamlit..." && \
streamlit run app.py
```

---

## 🌐 ACCÈS PORTFOLIO

**Une fois lancé:**

- **Local:** `http://localhost:8501`
- **Network:** `http://[Your-IP]:8501`
- **Docs API:** `/docs` (si endpoint FastAPI)

---

## 📌 PARAMÈTRES IMPORTANTS

| Paramètre | Défaut | Note |
|-----------|--------|------|
| **Port** | 8501 | Changer si conflit |
| **Server mode** | False | Auto-refresh activé |
| **Logger level** | info | Changer en debug si besoin |
| **Cache** | Enabled | `@st.cache_resource` utilisé |

---

## 🔐 PRODUCTION (Portfolio)

Pour un portfolio en production:

```bash
# 1. Vérifier tout fonctionne localement
streamlit run app.py

# 2. Builder Docker (optionnel)
docker build -t lstm-dashboard .
docker run -p 8501:8501 lstm-dashboard

# 3. Deployer sur Streamlit Cloud (gratuit)
# https://streamlit.io/cloud
# Git push → Auto-deploy
```

---

## 📞 RAPPORT DE STATUS

Avant de montrer le portfolio, lance ce check:

```bash
#!/bin/bash
echo "🔍 LSTM Dashboard Status Check"
echo "════════════════════════════════"

# Python
python --version

# Packages
python -c "import streamlit, torch, sklearn; print('✓ Packages OK')" || echo "✗ Packages missing"

# Model
ls models/checkpoints/latest.pt >/dev/null && echo "✓ Model found" || echo "✗ Model missing"

# Data
ls data/stock_data.csv >/dev/null && echo "✓ Data found" || echo "✗ Data missing"

echo "════════════════════════════════"
echo "✅ Ready to launch!"
```

Sauvegarde ce script comme `check.sh`:

```bash
chmod +x check.sh
./check.sh
```

---

## 🎬 ONE-LINER FINAL

**Copie-colle pour démarrer immédiatement:**

```bash
cd /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo/dashboard && streamlit run app.py --theme.base=dark
```

**Voilà!** 🚀
