"""Research package; heavy numerical and plotting modules are loaded on demand."""


def __getattr__(name):
    """Preserve historical ESN exports without importing plotting on CLI startup."""
    from importlib import import_module

    wsbm_esn = import_module(".wsbm_esn", __name__)

    try:
        return getattr(wsbm_esn, name)
    except AttributeError:
        raise AttributeError(f"module 'src' has no attribute {name!r}") from None
