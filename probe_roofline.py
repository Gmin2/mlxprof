"""Place a model's prefill and decode phases on this machine's roofline.

Per layer timing tells you where the time goes. It does not tell you whether the
time is well spent. The roofline does: it names the resource actually holding you
back, and how far you are from the ceiling for that resource.
"""

import argparse
import os
import time

import mlx.core as mx
from mlxprof import machine_ceiling, param_count, print_roofline, roofline, tree_bytes

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def main():
    ap = argparse.ArgumentParser(description="Roofline analysis of an mlx LLM")
    ap.add_argument("--model", default=os.environ.get("MLXPROF_MODEL", DEFAULT_MODEL))
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--prompt", default="Explain memory bandwidth and why it matters.")
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from mlx_lm import load, stream_generate

    print("measuring machine ceilings ...")
    ceiling = machine_ceiling()

    print(f"loading {args.model} ...")
    model, tok = load(args.model)
    params = param_count(model)
    wbytes = tree_bytes(model.parameters())

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)

    last = None
    for resp in stream_generate(model, tok, prompt, max_tokens=args.tokens):
        last = resp

    n_prompt, n_gen = last.prompt_tokens, last.generation_tokens
    prefill_s = n_prompt / last.prompt_tps
    decode_s = n_gen / last.generation_tps

    # 2 flops per parameter per token is the standard matmul-dominant estimate.
    prefill = roofline(2 * params * n_prompt, wbytes, prefill_s, ceiling)
    decode = roofline(2 * params * n_gen, wbytes * n_gen, decode_s, ceiling)

    print(f"\n  machine   {ceiling['device']}")
    print(f"            {ceiling['peak_gbs']:.0f} GB/s   {ceiling['peak_tflops']:.1f} TFLOP/s"
          f"   ridge {ceiling['ridge']:.0f} FLOP/byte")
    print(f"  model     {args.model}")
    print(f"            {params/1e6:.0f}M params, {wbytes/1024/1024:.0f} MB weights,"
          f" {wbytes*8/params:.2f} bits/param")
    print(f"  run       {n_prompt} prompt + {n_gen} generated tokens\n")

    print(f'  {"phase":<9} {"FLOP/byte":>9} {"bound":>8}  {"utilization":>16}')
    print_roofline("prefill", prefill)
    print_roofline("decode", decode)

    print(f"\n  prefill {last.prompt_tps:8.1f} tok/s   {prefill['achieved_tflops']:6.2f} TFLOP/s achieved")
    print(f"  decode  {last.generation_tps:8.1f} tok/s   {decode['achieved_gbs']:6.1f} GB/s achieved")
    print(f"  peak memory {last.peak_memory:.3f} GB")

    d = decode
    print(f"\n  verdict: decode is {d['bound']} bound at {d['utilization']*100:.0f}% {d['metric']}.")
    if d["bound"] == "memory":
        ceil_tps = last.generation_tps * d["headroom_x"]
        print(f"  every token reads all {wbytes/1024/1024:.0f} MB of weights, so at this machine's")
        print(f"  {ceiling['peak_gbs']:.0f} GB/s the ceiling is about {ceil_tps:.0f} tok/s. you are at {last.generation_tps:.0f}.")
        print(f"  a smaller quantisation moves that ceiling. more compute does not.")


if __name__ == "__main__":
    main()
