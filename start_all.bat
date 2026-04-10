@echo off
echo ========================================
echo    LSTM Demo - Demarrage du stack
echo ========================================

:: Activation conda et lancement metrics server
echo [1/3] Demarrage du metrics server...
start "Metrics Server" cmd /k "C:\Users\trist\miniconda3\Scripts\activate.bat lstm-env && cd C:\Users\trist\stephane\projets\lstm-demo && python metrics_server.py"

timeout /t 3 /nobreak >nul

:: Lancement Prometheus
echo [2/3] Demarrage de Prometheus...
start "Prometheus" cmd /k "C:\Users\trist\stephane\projets\prometheus\prometheus.exe --config.file=C:\Users\trist\stephane\projets\prometheus\prometheus.yml"

timeout /t 3 /nobreak >nul

:: Lancement Grafana
echo [3/3] Demarrage de Grafana...
net start grafana >nul 2>&1

timeout /t 3 /nobreak >nul

:: Lancement Streamlit
echo [4/4] Demarrage du dashboard Streamlit...
start "Streamlit" cmd /k "C:\Users\trist\miniconda3\Scripts\activate.bat lstm-env && cd C:\Users\trist\stephane\projets\lstm-demo && streamlit run dashboard\app.py"

echo.
echo ========================================
echo    Stack demarre !
echo    Streamlit  : http://localhost:8501
echo    Prometheus : http://localhost:9090
echo    Grafana    : http://localhost:3000
echo ========================================
echo.
pause
