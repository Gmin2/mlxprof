"""Command line front door.

    mlxprof bench --model <repo>     one screen: verdict, where time goes, memory
    mlxprof ceiling                  just the machine's two rooflines
"""

import argparse
import os
import time

import mlx.core as mx

from mlxprof.core import (
    LayerProfiler,
    leaf_total,
    machine_ceiling,
    mark_leaves,
    param_count,
    roofline,
    tree_bytes,
    working_set_mb,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MB = 1024 * 1024


def _bar(pct, width=36, ch="#"):
    return ch * max(0, min(width, int(round(pct / 100 * width))))


def cmd_ceiling(args):
    c = machine_ceiling()
    print(f"\n  {c['device']}")
    print(f"    memory bandwidth   {c['peak_gbs']:8.1f} GB/s      (measured)")
    print(f"    matmul throughput  {c['peak_tflops']:8.2f} TFLOP/s   (measured)")
    print(f"    ridge point        {c['ridge']:8.1f} FLOP/byte")
    print(f"    unified memory     {working_set_mb():8.0f} MB recommended working set\n")
    print("  below the ridge point you are memory bound, above it you are compute bound.\n")


def _layer_breakdown(model, token_id, seq=128):
    """Profile a forward pass long enough that real work dominates the forced syncs.

    A single token decode step has layers that run for microseconds, so the eval we
    force at each boundary costs more than the layer does and the breakdown becomes
    a measurement of our own instrumentation. A longer sequence fixes the ratio."""
    x = mx.broadcast_to(mx.array([[token_id]]), (1, seq))

    def run():
        t0 = time.perf_counter()
        mx.eval(model(x))
        return (time.perf_counter() - t0) * 1000

    for _ in range(3):
        run()
    clean = sorted(run() for _ in range(7))[3]

    prof = LayerProfiler().attach(model)
    run()
    prof.detach()
    records = mark_leaves(prof.records)

    agg = {}
    for r in records:
        if r["leaf"]:
            a = agg.setdefault(r["kind"], 0.0)
            agg[r["kind"]] = a + r["ms"]
    total = sum(agg.values()) or 1.0
    rows = sorted(agg.items(), key=lambda kv: -kv[1])
    return rows, total, clean, leaf_total(records)


def cmd_bench(args):
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("measuring machine ceilings ...")
    ceiling = machine_ceiling()

    print(f"loading {args.model} ...")
    from mlx_lm import load, stream_generate

    model, tok = load(args.model)
    params = param_count(model)
    wbytes = tree_bytes(model.parameters())

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    # prompt_tps on a cold call is ~10x low, it is measuring kernel compilation.
    for _ in stream_generate(model, tok, prompt, max_tokens=4):
        pass

    best = None
    for _ in range(args.trials):
        last = None
        for resp in stream_generate(model, tok, prompt, max_tokens=args.tokens):
            last = resp
        if best is None or last.generation_tps > best.generation_tps:
            best = last
    last = best

    n_p, n_g = last.prompt_tokens, last.generation_tokens
    prefill = roofline(2 * params * n_p, wbytes, n_p / last.prompt_tps, ceiling)
    decode = roofline(2 * params * n_g, wbytes * n_g, n_g / last.generation_tps, ceiling)

    rows, layer_total, clean_ms, leaf_ms = _layer_breakdown(model, last.token, args.seq)
    inflation = (leaf_ms / clean_ms - 1) * 100 if clean_ms else 0.0
    ws = working_set_mb()

    print(f"\n  machine    {ceiling['device']}, {ws:.0f} MB working set,"
          f" {ceiling['peak_gbs']:.0f} GB/s, {ceiling['peak_tflops']:.1f} TFLOP/s")
    print(f"  model      {args.model}")
    print(f"             {params/1e6:.0f}M params, {wbytes/MB:.0f} MB weights,"
          f" {wbytes*8/params:.2f} bits/param")
    print(f"  run        {n_p} prompt + {n_g} generated tokens\n")

    print(f'  {"":9} {"tok/s":>9} {"bound":>9} {"intensity":>11}   utilization')
    for name, r, tps in (("prefill", prefill, last.prompt_tps), ("decode", decode, last.generation_tps)):
        pct = r["utilization"] * 100
        print(f'  {name:<9} {tps:>9.1f} {r["bound"]:>9} {r["intensity"]:>8.1f} F/B'
              f'   {pct:5.1f}% {r["metric"]}  {_bar(pct)}')

    print(f"\n  where a {args.seq} token forward pass spends its time")
    for kind, ms in rows[:6]:
        pct = ms / layer_total * 100
        print(f"    {kind:<14} {pct:5.1f}%   {_bar(pct, 32)}")

    kv = max(0.0, last.peak_memory * 1024 - wbytes / MB)
    print(f"\n  memory     weights {wbytes/MB:.0f} MB   peak {last.peak_memory*1024:.0f} MB"
          f"   {last.peak_memory*1024/ws*100:.1f}% of working set")
    warn = "  <- too high to trust the split" if inflation > 60 else ""
    print(f"  accuracy   per layer numbers inflated ~{inflation:.0f}%, totals are exact{warn}")

    d = decode
    print(f"\n  VERDICT   decode is {d['bound']} bound at {d['utilization']*100:.0f}% {d['metric']}")
    if d["bound"] == "memory":
        ceil_tps = last.generation_tps * d["headroom_x"]
        print(f"            every token reads all {wbytes/MB:.0f} MB of weights. at"
              f" {ceiling['peak_gbs']:.0f} GB/s the ceiling is ~{ceil_tps:.0f} tok/s,"
              f" you are at {last.generation_tps:.0f}.")
        print(f"            a smaller quantisation raises that ceiling. more compute does not.\n")
    else:
        print(f"            {d['headroom_x']:.1f}x below the compute roofline at this intensity.\n")


def main():
    ap = argparse.ArgumentParser(prog="mlxprof", description="Profile mlx models on Apple silicon")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="one screen: verdict, where time goes, memory")
    b.add_argument("--model", default=os.environ.get("MLXPROF_MODEL", DEFAULT_MODEL))
    b.add_argument("--prompt", default="Explain memory bandwidth and why it matters.")
    b.add_argument("--tokens", type=int, default=96)
    b.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    b.add_argument("--seq", type=int, default=128, help="sequence length for the layer breakdown")
    b.add_argument("--trials", type=int, default=3, help="measured runs after warmup, best is kept")
    b.set_defaults(fn=cmd_bench)

    c = sub.add_parser("ceiling", help="measure this machine's rooflines")
    c.set_defaults(fn=cmd_ceiling)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
