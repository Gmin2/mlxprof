"""Step by step tour of what mlxprof does and why it has to work this way.

Run one step at a time:   python walkthrough.py 1
Or all of them:           python walkthrough.py all
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn

STEPS = {}


def step(n, title):
    def deco(fn):
        STEPS[n] = (title, fn)
        return fn

    return deco


def header(n):
    title = STEPS[n][0]
    print(f"\n{'=' * 68}\nSTEP {n}. {title}\n{'=' * 68}")


@step(1, "mlx is lazy, so naive timing measures nothing")
def s1():
    layer = nn.Linear(4096, 4096)
    x = mx.random.normal((1, 4096))
    mx.eval(layer.parameters(), x)
    for _ in range(3):
        mx.eval(layer(x))

    t0 = time.perf_counter()
    y = layer(x)
    t1 = time.perf_counter()
    mx.eval(y)
    t2 = time.perf_counter()

    print(f"  calling the layer     {(t1 - t0) * 1000:8.4f} ms   <- builds a graph node, no math")
    print(f"  mx.eval on the result {(t2 - t1) * 1000:8.4f} ms   <- the actual matmul")
    print(f"  ratio                 {(t2 - t1) / (t1 - t0):8.0f}x")
    print("\n  a profiler that wraps the call with a timer reports the first number.")
    print("  to get a real one we must force mx.eval at every layer boundary.")


@step(2, "you cannot hook mlx modules the way you hook pytorch")
def s2():
    lin = nn.Linear(8, 8)
    x = mx.zeros((1, 8))

    lin.__call__ = lambda _x: "INTERCEPTED"
    got = lin(x)
    print(f"  patching the instance -> {type(got).__name__}   (patch ignored)")

    cls = type(lin)
    lin.__class__ = type("P", (cls,), {"__call__": lambda self, _x: "INTERCEPTED"})
    print(f"  swapping the class    -> {lin(x)}")
    print("\n  python looks up dunder methods on the type, never the instance.")
    print("  nn.Module has no register_forward_hook, so the class swap is the way in.")


@step(3, "the interception, in eight lines")
def s3():
    model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
    x = mx.random.normal((1, 512))
    mx.eval(model.parameters(), x)
    for _ in range(3):
        mx.eval(model(x))
    log = []

    for name, mod in model.named_modules():
        if not name:
            continue
        cls = type(mod)
        orig = cls.__call__

        def timed(self, *a, _n=name, _o=orig, **k):
            t0 = time.perf_counter()
            out = _o(self, *a, **k)
            mx.eval(out)
            log.append((_n, (time.perf_counter() - t0) * 1000))
            return out

        mod.__class__ = type("P_" + cls.__name__, (cls,), {"__call__": timed})

    mx.eval(model(x))
    for n, ms in log:
        print(f"  {n:<12} {ms:7.3f} ms")
    print("\n  that is the whole mechanism. mlxprof adds memory, nesting and reporting.")
    print("  note the three warmup runs above the loop. without them the first eval")
    print("  absorbs kernel compilation and lands on whichever layer happened to be first.")


@step(4, "what forcing eval costs you")
def s4():
    from mlxprof import LayerProfiler, leaf_total, mark_leaves, print_by_kind

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.up = nn.Linear(d, d * 4)
            self.act = nn.GELU()
            self.down = nn.Linear(d * 4, d)

        def __call__(self, x):
            return x + self.down(self.act(self.up(self.norm(x))))

    class Stack(nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.blocks = [Block(d) for _ in range(n)]

        def __call__(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    model = Stack(2048, 8)
    x = mx.random.normal((1, 128, 2048))
    mx.eval(model.parameters(), x)

    def run():
        t0 = time.perf_counter()
        mx.eval(model(x))
        return (time.perf_counter() - t0) * 1000

    for _ in range(5):
        run()
    clean = sorted(run() for _ in range(9))[4]

    prof = LayerProfiler().attach(model)
    dirty = run()
    prof.detach()
    records = mark_leaves(prof.records)
    leaves = leaf_total(records)

    print_by_kind(records)
    print(f"\n  unprofiled   {clean:7.2f} ms   <- the truth")
    print(f"  profiled     {dirty:7.2f} ms   <- {dirty / clean:.2f}x, cost of measuring")
    print(f"  leaf sum     {leaves:7.2f} ms   <- {leaves / clean * 100:.0f}% of truth")
    print("\n  the SHAPE is right, Linear really does dominate. every absolute number is inflated.")


@step(5, "where the memory actually goes")
def s5():
    from mlxprof import LayerProfiler, mark_leaves, print_memory_timeline, weight_mb, working_set_mb

    layers = []
    for _ in range(3):
        layers += [nn.Linear(2048, 8192), nn.GELU(), nn.Linear(8192, 2048)]
    model = nn.Sequential(*layers)
    x = mx.random.normal((1, 2048, 2048))
    mx.eval(model.parameters(), x)

    w, ws = weight_mb(model), working_set_mb()
    mx.clear_cache()
    mx.reset_peak_memory()

    prof = LayerProfiler().attach(model)
    mx.eval(model(x))
    prof.detach()

    print(f"  weights {w:.0f} MB, working set limit {ws:.0f} MB\n")
    print_memory_timeline(mark_leaves(prof.records), w, ws)
    print("\n  watch the cache column. when activations drop, cache rises by the same amount.")
    print("  freed memory goes to mlx's pool, not back to the os. that is why 'freeing' does nothing.")


@step(6, "on a real model")
def s6():
    print("  run:  python probe_whisper.py --hf-home <your hf cache>")
    print("  32 layer whisper encoder, 323 instrumented calls.")
    print("  the inflation drops from ~190% to ~125% because bigger layers")
    print("  amortise the forced sync over more real work.")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = sorted(STEPS) if arg == "all" else [int(arg)]
    for k in keys:
        header(k)
        STEPS[k][1]()


if __name__ == "__main__":
    main()
