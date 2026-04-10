#!/bin/bash
# Démarrage du metrics server en arrière-plan
python metrics_server.py &

# Démarrage Streamlit
exec streamlit run dashboard/app.py
