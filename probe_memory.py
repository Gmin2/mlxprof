import mlx.core as mx
import mlx.nn as nn

from mlxprof import (
    LayerProfiler,
    mark_leaves,
    print_memory_timeline,
    weight_mb,
    working_set_mb,
)

DIM = 2048
BLOCKS = 6
SEQ = 2048


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

    def __call__(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def main():
    model = Stack(DIM, BLOCKS)
    x = mx.random.normal((1, SEQ, DIM))
    mx.eval(model.parameters(), x)

    w = weight_mb(model)
    ws = working_set_mb()

    mx.clear_cache()
    mx.reset_peak_memory()

    prof = LayerProfiler().attach(model)
    mx.eval(model(x))
    prof.detach()

    records = mark_leaves(prof.records)

    print(f"device: {mx.device_info()['device_name']}")
    print(f"model:  {BLOCKS} blocks, dim {DIM}, input {tuple(x.shape)}")
    print(f"weights: {w:.1f} MB   input: {x.size * 4 / 1024 / 1024:.1f} MB   recommended working set: {ws:.0f} MB")
    print()
    print_memory_timeline(records, w, ws)
    print()
    print(f"  peak active: {mx.get_peak_memory() / 1024 / 1024:.1f} MB ({mx.get_peak_memory() / 1024 / 1024 / ws * 100:.1f}% of working set)")
    print(f"  mlx buffer cache at end: {mx.get_cache_memory() / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
