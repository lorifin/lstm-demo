"""Prometheus metrics HTTP server for LSTM Dashboard."""

from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Counters
predictions_total = Counter(
    'lstm_predictions_total',
    'Total number of predictions made',
    ['ticker']
)

training_total = Counter(
    'lstm_training_total',
    'Total number of training runs',
    ['ticker']
)

errors_total = Counter(
    'lstm_errors_total',
    'Total number of errors',
    ['error_type']
)

# Histograms
prediction_duration = Histogram(
    'lstm_prediction_duration_seconds',
    'Time spent making predictions',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

training_duration = Histogram(
    'lstm_training_duration_seconds',
    'Time spent training models',
    buckets=[10, 30, 60, 120, 300, 600]
)

# Gauges
model_mae = Gauge(
    'lstm_model_mae',
    'Mean Absolute Error of latest model',
    ['ticker']
)

model_rmse = Gauge(
    'lstm_model_rmse',
    'Root Mean Square Error of latest model',
    ['ticker']
)

model_mape = Gauge(
    'lstm_model_mape',
    'Mean Absolute Percentage Error of latest model',
    ['ticker']
)

model_r2 = Gauge(
    'lstm_model_r2',
    'R² score of latest model',
    ['ticker']
)


def start_metrics_server(port=8099):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    print(f"✓ Prometheus metrics server started on http://localhost:{port}/metrics")


def record_prediction(ticker: str, duration: float):
    """Record a prediction."""
    predictions_total.labels(ticker=ticker).inc()
    prediction_duration.observe(duration)


def record_training(ticker: str, duration: float, mae: float, rmse: float, mape: float, r2: float = 0.0):
    """Record a training run."""
    training_total.labels(ticker=ticker).inc()
    training_duration.observe(duration)
    model_mae.labels(ticker=ticker).set(mae)
    model_rmse.labels(ticker=ticker).set(rmse)
    model_mape.labels(ticker=ticker).set(mape)
    model_r2.labels(ticker=ticker).set(r2)


def record_error(error_type: str):
    """Record an error."""
    errors_total.labels(error_type=error_type).inc()


if __name__ == "__main__":
    start_metrics_server()
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n✓ Prometheus metrics server stopped")
