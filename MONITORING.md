# 📊 LSTM Dashboard Prometheus Monitoring

Complete monitoring stack for your LSTM Time Series Dashboard with Prometheus metrics.

## 🚀 Quick Start

### 1️⃣ Install Prometheus (one-time only)

```bash
# macOS
brew install prometheus

# Linux
sudo apt-get install prometheus

# Or from source
https://prometheus.io/download/
```

### 2️⃣ Start Monitoring Stack

```bash
cd /Users/stephanekrebs/bureau/OPENCLAW/lstm-demo

# Make script executable
chmod +x start_monitoring.sh

# Start all services
bash start_monitoring.sh
```

or separately:

```bash
# Terminal 1: Metrics Server
python metrics_server.py

# Terminal 2: Prometheus
prometheus --config.file=prometheus.yml

# Terminal 3: Streamlit Dashboard
cd dashboard && streamlit run app.py
```

### 3️⃣ Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit** | `http://localhost:8501` | Dashboard UI |
| **Prometheus** | `http://localhost:9090` | Metrics database |
| **Metrics** | `http://localhost:8099/metrics` | Prometheus endpoint |

---

## 📈 Available Metrics

### Counters (Total Counts)

```
lstm_predictions_total{ticker="MC.PA"}        # Total predictions made
lstm_training_total{ticker="MC.PA"}           # Total training runs
lstm_errors_total{error_type="training"}      # Total errors
```

### Histograms (Duration/Latency)

```
lstm_prediction_duration_seconds              # Prediction latency
lstm_training_duration_seconds                 # Training time
```

### Gauges (Current Values)

```
lstm_model_mae{ticker="MC.PA"}                # Mean Absolute Error
lstm_model_rmse{ticker="MC.PA"}               # Root Mean Square Error
lstm_model_mape{ticker="MC.PA"}               # Mean Absolute % Error
lstm_model_r2{ticker="MC.PA"}                 # R² Score
```

---

## 🔍 Prometheus Queries

### Prediction Rate (per 5 minutes)

```promql
rate(lstm_predictions_total[5m])
```

### Average Prediction Duration

```promql
rate(lstm_prediction_duration_seconds_sum[5m]) / rate(lstm_prediction_duration_seconds_count[5m])
```

### Error Rate

```promql
rate(lstm_errors_total[5m])
```

### Latest Model Performance

```promql
lstm_model_mae
lstm_model_rmse
lstm_model_mape
```

### Training Duration for Latest Run

```promql
lstm_training_duration_seconds
```

---

## 📊 Create Prometheus Dashboard

In Prometheus UI (`http://localhost:9090`):

1. Click "Graph"
2. Enter a query (from examples above)
3. Click "Execute"
4. Visualize metrics in real-time

Example dashboard queries:

```
# Row 1: Prediction Stats
lstm_predictions_total
rate(lstm_predictions_total[5m])

# Row 2: Model Performance
lstm_model_mae
lstm_model_rmse

# Row 3: Latency
lstm_prediction_duration_seconds
```

---

## 🔧 Customize Metrics

Edit `metrics_server.py` to add new metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Add custom metric
my_metric = Counter('my_lstm_custom', 'My custom metric', ['label'])

# Use in code
my_metric.labels(label='value').inc()
```

---

## 📋 Prometheus Configuration

Config file: `prometheus.yml`

- Scrape interval: 15 seconds
- Metrics path: `/metrics`
- Targets: `localhost:8099` (metrics server)

To update scrape interval:

```yaml
global:
  scrape_interval: 10s  # Change this
```

Then restart Prometheus.

---

## 🚨 Troubleshooting

### Metrics not showing in Prometheus

1. Check metrics server is running:
   ```bash
   curl http://localhost:8099/metrics
   ```

2. Check Prometheus targets:
   Go to `http://localhost:9090/targets`
   Should show "UP" for the job

3. Restart Prometheus:
   ```bash
   # Kill and restart
   pkill prometheus
   prometheus --config.file=prometheus.yml
   ```

### Port already in use

```bash
# Find what's using the port
lsof -i :8099  # Metrics server
lsof -i :9090  # Prometheus
lsof -i :8501  # Streamlit

# Kill the process
kill -9 <PID>
```

---

## 📚 Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)

---

## 🎯 Next Steps

1. **Add Grafana**: Create beautiful dashboards
   ```bash
   brew install grafana
   brew services start grafana
   ```

2. **Set up AlertManager**: Get alerts on anomalies

3. **Store metrics long-term**: Use Prometheus remote storage

---

**Your monitoring stack is live!** 🚀

Check `http://localhost:9090` to start monitoring your LSTM dashboard.
