"""Profile an autoregressive decode loop rather than a single forward pass.

A forward pass has no kv cache and runs once. Generation runs the decoder once
per token, carrying a cache that grows every step. That is where both the time
and the memory actually go.
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np

from mlxprof import tree_mb, working_set_mb

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"


def main():
    ap = argparse.ArgumentParser(description="Profile the whisper decode loop")
    ap.add_argument("--repo", default=os.environ.get("MLXPROF_WHISPER_REPO", DEFAULT_REPO))
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--every", type=int, default=4, help="print every Nth step")
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from mlx_whisper import load_models
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim

    print(f"loading {args.repo} ...")
    model = load_models.load_model(args.repo)
    dims = model.dims

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 10) * 0.05).astype(np.float32)
    mel = log_mel_spectrogram(pad_or_trim(mx.array(audio), 480000), n_mels=dims.n_mels)[None]
    mx.eval(model.parameters(), mel)

    t0 = time.perf_counter()
    audio_features = model.encoder(mel)
    mx.eval(audio_features)
    encode_cold_ms = (time.perf_counter() - t0) * 1000

    for _ in range(2):
        mx.eval(model.encoder(mel))
    t0 = time.perf_counter()
    mx.eval(model.encoder(mel))
    encode_ms = (time.perf_counter() - t0) * 1000

    mx.clear_cache()
    mx.reset_peak_memory()
    base_active = mx.get_active_memory() / 1024 / 1024
    ws = working_set_mb()

    tokens = mx.array([[dims.n_vocab - 10]])
    kv_cache = None
    rows = []

    for step in range(1, args.steps + 1):
        t = time.perf_counter()
        logits, kv_cache, _ = model.decoder(tokens, audio_features, kv_cache=kv_cache)
        mx.eval(logits, kv_cache)
        ms = (time.perf_counter() - t) * 1000

        self_mb = sum(tree_mb(b[0]) for b in kv_cache)
        cross_mb = sum(tree_mb(b[1]) for b in kv_cache)
        rows.append((step, ms, self_mb, cross_mb, mx.get_active_memory() / 1024 / 1024))

        tokens = mx.argmax(logits[:, -1], axis=-1)[None]

    print(f"\ndevice: {mx.device_info()['device_name']}")
    print(f"model:  whisper, {dims.n_text_layer} decoder layers, width {dims.n_text_state}")
    print(f"encoder: {encode_cold_ms:.1f} ms cold, {encode_ms:.1f} ms warm")
    print(f"decoder: {args.steps} steps\n")

    print(f'  {"step":>5} {"ms":>8} {"kv self":>9} {"kv cross":>9} {"active":>10}')
    for step, ms, s_mb, c_mb, act in rows:
        if step % args.every == 0 or step == 1:
            print(f"  {step:>5} {ms:>8.2f} {s_mb:>8.2f}M {c_mb:>8.2f}M {act:>9.1f}M")

    warm = rows[2:]
    total_decode = sum(r[1] for r in rows)
    first, last = rows[0], rows[-1]
    per_tok_kv = (last[2] - first[2]) / (last[0] - first[0]) if len(rows) > 1 else 0

    print(f"\n  encode warm         {encode_ms:8.1f} ms  (one pass, {encode_cold_ms:.0f} ms cold)")
    print(f"  decode {args.steps:>3} steps    {total_decode:8.1f} ms  ({total_decode / args.steps:.2f} ms/step)")
    print(f"  first step          {first[1]:8.2f} ms  <- includes warmup")
    print(f"  median warm step    {sorted(r[1] for r in warm)[len(warm) // 2]:8.2f} ms")
    print(f"\n  kv self  grows      {per_tok_kv * 1024:8.1f} KB per token  ({first[2]:.2f}M -> {last[2]:.2f}M)")
    print(f"  kv cross is fixed   {first[3]:8.2f} MB  (encoder output, computed once)")
    print(f"  active memory       {base_active:.1f}M -> {last[4]:.1f}M   peak {mx.get_peak_memory() / 1024 / 1024:.1f}M ({mx.get_peak_memory() / 1024 / 1024 / ws * 100:.1f}% of working set)")

    proj = per_tok_kv * 448
    print(f"\n  at whisper's 448 token limit, kv self would reach {proj:.1f} MB")


if __name__ == "__main__":
    main()
