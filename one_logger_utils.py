class OneLoggerUtils:
    """No-op fallback for NVIDIA internal One Logger hooks.

    LongLive only uses these hooks when one_logger is enabled. This fallback
    keeps training entrypoints importable in public environments where the
    internal package is unavailable.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if name.startswith("on_"):
            return self._noop
        raise AttributeError(name)

    def _noop(self, *args, **kwargs):
        return None
