"""Prometheus metrics for LSTM Dashboard monitoring."""

from prometheus_client import Counter, Histogram, Gauge
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

data_loads_total = Counter(
    'lstm_data_loads_total',
    'Total number of data loads',
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
    buckets=[10, 30, 60, 120, 300]
)

data_load_duration = Histogram(
    'lstm_data_load_duration_seconds',
    'Time spent loading data',
    buckets=[1, 5, 10, 30]
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

# Decorators for timing
def track_prediction_time(ticker):
    """Decorator to track prediction time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                predictions_total.labels(ticker=ticker).inc()
                return result
            except Exception as e:
                errors_total.labels(error_type='prediction').inc()
                raise
            finally:
                duration = time.time() - start
                prediction_duration.observe(duration)
        return wrapper
    return decorator


def track_training_time(ticker):
    """Decorator to track training time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                training_total.labels(ticker=ticker).inc()
                return result
            except Exception as e:
                errors_total.labels(error_type='training').inc()
                raise
            finally:
                duration = time.time() - start
                training_duration.observe(duration)
        return wrapper
    return decorator


def track_data_load_time(ticker):
    """Decorator to track data loading time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                data_loads_total.labels(ticker=ticker).inc()
                return result
            except Exception as e:
                errors_total.labels(error_type='data_load').inc()
                raise
            finally:
                duration = time.time() - start
                data_load_duration.observe(duration)
        return wrapper
    return decorator
