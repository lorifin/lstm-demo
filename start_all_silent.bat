@echo off
:: Demarrage Metrics Server
start /min "" cmd /c "C:\Users\trist\miniconda3\Scripts\activate.bat lstm-env && cd C:\Users\trist\stephane\projets\lstm-demo && python metrics_server.py"

timeout /t 3 /nobreak >nul

:: Demarrage Prometheus
start /min "" "C:\Users\trist\stephane\projets\prometheus\prometheus.exe" --config.file="C:\Users\trist\stephane\projets\prometheus\prometheus.yml"

timeout /t 3 /nobreak >nul

:: Demarrage Grafana
net start grafana >nul 2>&1

timeout /t 3 /nobreak >nul

:: Demarrage Streamlit
start /min "" cmd /c "C:\Users\trist\miniconda3\Scripts\activate.bat lstm-env && cd C:\Users\trist\stephane\projets\lstm-demo && streamlit run dashboard\app.py"
