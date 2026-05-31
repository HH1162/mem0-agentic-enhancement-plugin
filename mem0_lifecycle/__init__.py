"""Mem0 Memory Enhancement via Forgetting Curve Model.

A production-tested memory lifecycle system that makes Mem0 behave like
human memory — frequently accessed information persists, unused information
naturally decays.

This package implements an exponential decay model inspired by the
Ebbinghaus forgetting curve, sitting as a bridge layer between your
application and Mem0. No core Mem0 modification required.

Reference: https://github.com/HH1162/mem0-lifecycle
Issue: https://github.com/mem0ai/mem0/issues/5330
"""

__version__ = "0.1.0"

# Core components - import lazily to avoid circular dependencies
def __getattr__(name):
    if name == "Mem0LifecycleServer":
        from .server import Mem0LifecycleServer
        return Mem0LifecycleServer
    elif name == "compute_weighted_score":
        from .decay import compute_weighted_score
        return compute_weighted_score
    elif name == "should_cleanup":
        from .decay import should_cleanup
        return should_cleanup
    elif name == "is_grace_protected":
        from .decay import is_grace_protected
        return is_grace_protected
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
