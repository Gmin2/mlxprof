"""Instrument every mlx module class, without being handed a model.

Patching nn.Module.__call__ once does not work: subclasses define their own
__call__ and never reach the base. So we wrap every class that defines one, and
hook __init_subclass__ so classes defined later (mlx_lm builds its model classes
at load time) get wrapped as they appear.
"""

import threading
import time

import mlx.core as mx
import mlx.nn as nn

MB = 1024 * 1024


class AutoProfiler:
    """Whole-process instrumentation.

    Two modes, and the difference is honesty. By default we do not force
    evaluation, so a run's total time is real and per layer times are not
    recorded at all, because without a sync they would be graph construction.
    In detail mode we force eval at every boundary, which gives per layer numbers
    and inflates the run by 1.3x or more. The run says which mode produced it.
    """

    def __init__(self, detail=False, keep=50):
        self.detail = detail
        self.keep = keep
        self.runs = []
        self._wrapped = set()
        self._installed = False
        self._depth = 0
        self._calls = []
        self._t0 = None
        self._next_id = 0
        self._lock = threading.Lock()

    def install(self):
        if self._installed:
            return self
        self._walk(nn.Module)
        prof = self

        def hook(cls, **kw):
            prof._wrap(cls)

        nn.Module.__init_subclass__ = classmethod(hook)
        self._installed = True
        return self

    def _walk(self, cls):
        self._wrap(cls)
        for sub in cls.__subclasses__():
            self._walk(sub)

    def _wrap(self, cls):
        if cls in self._wrapped or "__call__" not in cls.__dict__:
            return
        orig = cls.__dict__["__call__"]
        prof = self

        def timed(inner_self, *args, **kwargs):
            if prof._depth == 0:
                prof._begin()
            slot = len(prof._calls)
            prof._calls.append(None)
            depth = prof._depth
            prof._depth += 1
            t = time.perf_counter()
            try:
                out = orig(inner_self, *args, **kwargs)
                if prof.detail:
                    try:
                        mx.eval(out)
                    except Exception:
                        pass
            finally:
                prof._depth -= 1
            prof._calls[slot] = {
                "name": cls.__name__,
                "depth": depth,
                "ms": (time.perf_counter() - t) * 1000 if prof.detail else None,
            }
            if prof._depth == 0:
                prof._end(out)
            return out

        cls.__call__ = timed
        self._wrapped.add(cls)

    def _begin(self):
        self._calls = []
        mx.reset_peak_memory()
        self._t0 = time.perf_counter()

    def _end(self, out):
        try:
            mx.eval(out)
        except Exception:
            pass
        total = (time.perf_counter() - self._t0) * 1000
        kinds = {}
        for c in self._calls:
            if not c:
                continue
            k = kinds.setdefault(c["name"], {"calls": 0, "ms": 0.0})
            k["calls"] += 1
            if c["ms"] is not None:
                k["ms"] += c["ms"]
        self._next_id += 1
        run = {
            "id": self._next_id,
            "at": time.time(),
            "total_ms": round(total, 3),
            "calls": len(self._calls),
            "modules": len(kinds),
            "detail": self.detail,
            "peak_mb": round(mx.get_peak_memory() / MB, 1),
            "kinds": [
                {"name": n, "calls": v["calls"], "ms": round(v["ms"], 3)}
                for n, v in sorted(
                    kinds.items(),
                    key=(lambda kv: -kv[1]["ms"]) if self.detail else (lambda kv: -kv[1]["calls"]),
                )
            ],
        }
        with self._lock:
            self.runs.append(run)
            if len(self.runs) > self.keep:
                del self.runs[: len(self.runs) - self.keep]
        self._calls = []
