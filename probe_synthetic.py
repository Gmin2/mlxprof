import time

import mlx.core as mx
import mlx.nn as nn

from mlxprof import LayerProfiler, leaf_total, mark_leaves, print_by_kind, print_tree

DIM = 2048
BLOCKS = 8


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.down = nn.Linear(dim * 4, dim)

    def __call__(self, x):
        return x + self.down(self.act(self.up(self.norm(x))))


class Stack(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.blocks = [Block(dim) for _ in range(n)]
        self.out = nn.Linear(dim, dim)

    def __call__(self, x):
        for b in self.blocks:
            x = b(x)
        return self.out(x)


def timed_run(model, x):
    t0 = time.perf_counter()
    y = model(x)
    mx.eval(y)
    return (time.perf_counter() - t0) * 1000


def main():
    model = Stack(DIM, BLOCKS)
    x = mx.random.normal((1, 128, DIM))
    mx.eval(model.parameters(), x)

    for _ in range(5):
        timed_run(model, x)

    clean = sorted(timed_run(model, x) for _ in range(10))[len(range(10)) // 2]

    prof = LayerProfiler().attach(model)
    for _ in range(3):
        timed_run(model, x)
        prof.reset()
    dirty = timed_run(model, x)
    prof.detach()

    records = mark_leaves(prof.records)
    leaves = leaf_total(records)

    print(f"device: {mx.device_info()['device_name']}")
    print(f"model:  {BLOCKS} blocks, dim {DIM}, input {tuple(x.shape)}")
    print(f"modules instrumented: {len(prof._patched) if prof._patched else sum(1 for _ in model.named_modules()) - 1}")
    print()
    print("per layer (first 20, execution order)")
    print_tree(records, limit=20)
    print()
    print("by module kind (leaves only)")
    print_by_kind(records)
    print()
    print("reconciliation")
    print(f"  unprofiled end to end   {clean:8.3f} ms   <- the truth")
    print(f"  profiled end to end     {dirty:8.3f} ms   <- cost of measuring")
    print(f"  sum of leaf layers      {leaves:8.3f} ms   <- what a naive profiler reports")
    print()
    print(f"  observer overhead       {dirty - clean:8.3f} ms  ({dirty/clean:.2f}x slower)")
    print(f"  unaccounted in leaf sum {clean - leaves:8.3f} ms  ({leaves/clean*100:.1f}% of truth explained)")


if __name__ == "__main__":
    main()
