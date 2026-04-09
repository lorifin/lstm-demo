#!/bin/bash

# Script to start Prometheus for LSTM Dashboard monitoring

echo "🚀 Starting LSTM Dashboard Monitoring Stack"
echo "=========================================="
echo ""

# Check if prometheus installed
if ! command -v prometheus &> /dev/null; then
    echo "❌ Prometheus not installed!"
    echo "Install with: brew install prometheus"
    exit 1
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "📂 Project directory: $PROJECT_DIR"
echo ""

# Start metrics server in background
echo "🔧 Starting Prometheus metrics server..."
cd "$PROJECT_DIR"
python metrics_server.py &
METRICS_PID=$!
echo "   PID: $METRICS_PID"
echo "   URL: http://localhost:8099/metrics"
echo ""

# Start Prometheus
echo "📊 Starting Prometheus..."
prometheus --config.file="$PROJECT_DIR/prometheus.yml" --web.console.templates=/usr/local/etc/prometheus/consoles --web.console.libraries=/usr/local/etc/prometheus/console_libraries &
PROMETHEUS_PID=$!
echo "   PID: $PROMETHEUS_PID"
echo "   URL: http://localhost:9090"
echo ""

echo "=========================================="
echo "✅ Monitoring stack is running!"
echo ""
echo "📊 Prometheus Dashboard: http://localhost:9090"
echo "🔧 Metrics Endpoint: http://localhost:8099/metrics"
echo ""
echo "📈 Useful Prometheus Queries:"
echo "   - Rate of predictions: rate(lstm_predictions_total[5m])"
echo "   - Training duration: lstm_training_duration_seconds"
echo "   - Model MAE: lstm_model_mae"
echo "   - Error rate: rate(lstm_errors_total[5m])"
echo ""
echo "To stop: press Ctrl+C"
echo "=========================================="

# Wait for both processes
wait $METRICS_PID $PROMETHEUS_PID
