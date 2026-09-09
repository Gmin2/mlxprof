import time

import mlx.core as mx
from mlx.utils import tree_flatten

MB = 1024 * 1024


class LayerProfiler:
    """Times every submodule of an mlx model by forcing evaluation at each boundary.

    mlx is lazy, so calling a layer only builds graph nodes. To get a real number
    out of a layer we have to mx.eval its output, which also syncs and prevents
    fusion across the boundary. That is why profiled totals do not match the
    unprofiled run. Memory is sampled at the same boundary, where it is exact.
    """

    def __init__(self):
        self.records = []
        self._patched = []
        self._depth = 0

    def attach(self, model):
        for name, mod in model.named_modules():
            if name:
                self._patch(name, mod)
        return self

    def detach(self):
        for mod, cls in self._patched:
            mod.__class__ = cls
        self._patched.clear()

    def reset(self):
        self.records.clear()

    def _patch(self, name, mod):
        cls = type(mod)
        orig = cls.__call__
        prof = self

        def timed(inner_self, *args, **kwargs):
            slot = len(prof.records)
            prof.records.append(None)
            depth = prof._depth
            prof._depth += 1
            t0 = time.perf_counter()
            try:
                out = orig(inner_self, *args, **kwargs)
                try:
                    mx.eval(out)
                except Exception:
                    pass
            finally:
                prof._depth -= 1
            prof.records[slot] = {
                "name": name,
                "kind": cls.__name__,
                "depth": depth,
                "ms": (time.perf_counter() - t0) * 1000,
                "active_mb": mx.get_active_memory() / MB,
                "cache_mb": mx.get_cache_memory() / MB,
                "peak_mb": mx.get_peak_memory() / MB,
            }
            return out

        self._patched.append((mod, cls))
        mod.__class__ = type("Profiled_" + cls.__name__, (cls,), {"__call__": timed})


def weight_mb(model):
    return sum(p.size * p.dtype.size for _, p in tree_flatten(model.parameters())) / MB


def tree_bytes(obj):
    """Bytes of every mx.array reachable in a nested structure. Shared arrays count once."""
    seen = {}

    def walk(o):
        if isinstance(o, mx.array):
            seen[id(o)] = o.size * o.dtype.size
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(obj)
    return sum(seen.values())


def tree_mb(obj):
    return tree_bytes(obj) / MB


def working_set_mb():
    return mx.device_info()["max_recommended_working_set_size"] / MB


def mark_leaves(records):
    for i, r in enumerate(records):
        nxt = records[i + 1] if i + 1 < len(records) else None
        r["leaf"] = nxt is None or nxt["depth"] <= r["depth"]
    return records


def leaf_total(records):
    return sum(r["ms"] for r in records if r["leaf"])


def print_tree(records, limit=None):
    rows = records if limit is None else records[:limit]
    for r in rows:
        indent = "  " * r["depth"]
        tag = "" if r["leaf"] else "  (parent)"
        print(f'  {r["ms"]:8.3f} ms  {indent}{r["name"]} [{r["kind"]}]{tag}')
    if limit is not None and len(records) > limit:
        print(f"  ... {len(records) - limit} more")


def print_by_kind(records):
    agg = {}
    for r in records:
        if not r["leaf"]:
            continue
        a = agg.setdefault(r["kind"], {"n": 0, "ms": 0.0})
        a["n"] += 1
        a["ms"] += r["ms"]
    total = sum(a["ms"] for a in agg.values()) or 1.0
    print(f'  {"kind":<20} {"calls":>6} {"total ms":>10} {"share":>7}')
    for kind, a in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
        print(f'  {kind:<20} {a["n"]:>6} {a["ms"]:>10.3f} {a["ms"]/total*100:>6.1f}%')


def print_memory_timeline(records, weights, limit, width=40, only_leaves=True):
    rows = [r for r in records if r["leaf"]] if only_leaves else list(records)
    if not rows:
        return
    top = max(r["active_mb"] for r in rows)
    scale = max(top, weights) or 1.0
    wbar = int(weights / scale * width)
    print(f'  {"layer":<24} {"active":>9} {"activs":>9} {"cache":>8}  {"":<{width}} {"%ws":>6}')
    for r in rows:
        acts = r["active_mb"] - weights
        n = int(r["active_mb"] / scale * width)
        bar = "=" * min(wbar, n) + "+" * max(0, n - wbar)
        print(
            f'  {r["name"]:<24} {r["active_mb"]:>8.1f}M {acts:>8.1f}M {r["cache_mb"]:>7.1f}M'
            f'  {bar:<{width}} {r["active_mb"]/limit*100:>5.1f}%'
        )
    print(f"  legend: '=' weights ({weights:.1f}M)  '+' activations   %ws = share of recommended working set")
