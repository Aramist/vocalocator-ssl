from torch.profiler import record_function


def record(func):
    def wrapper(*args, **kwargs):
        if hasattr(func, "__self__"):
            name = f"{func.__self__.__class__.__name__}.{func.__name__}"
        else:
            name = func.__name__
        with record_function(name):
            return func(*args, **kwargs)

    return wrapper
