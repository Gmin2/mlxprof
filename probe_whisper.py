import argparse
import os
import time

import mlx.core as mx
import numpy as np

from mlxprof import (
    LayerProfiler,
    leaf_total,
    mark_leaves,
    print_by_kind,
    print_memory_timeline,
    weight_mb,
    working_set_mb,
)

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000


def load_audio(path):
    if path is None:
        rng = np.random.default_rng(0)
        return rng.standard_normal(SAMPLE_RATE * 10).astype(np.float32) * 0.05, True

    import soundfile as sf

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio
        )
    return audio.astype(np.float32), False


def main():
    ap = argparse.ArgumentParser(description="Profile an mlx-whisper encoder layer by layer")
    ap.add_argument("--repo", default=os.environ.get("MLXPROF_WHISPER_REPO", DEFAULT_REPO))
    ap.add_argument("--wav", default=os.environ.get("MLXPROF_WAV"))
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--rows", type=int, default=40, help="memory rows to print")
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from mlx_whisper import load_models
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim

    audio, synthetic = load_audio(args.wav)

    print(f"loading {args.repo} ...")
    model = load_models.load_model(args.repo)
    dims = model.dims
    mel = log_mel_spectrogram(pad_or_trim(mx.array(audio), 480000), n_mels=dims.n_mels)[None]
    mx.eval(model.parameters(), mel)

    enc = model.encoder
    w = weight_mb(enc)
    ws = working_set_mb()

    def run():
        t0 = time.perf_counter()
        mx.eval(enc(mel))
        return (time.perf_counter() - t0) * 1000

    for _ in range(3):
        run()
    clean = sorted(run() for _ in range(5))[2]

    mx.clear_cache()
    mx.reset_peak_memory()
    prof = LayerProfiler().attach(enc)
    dirty = run()
    prof.detach()

    records = mark_leaves(prof.records)
    leaves = leaf_total(records)

    src = "synthetic noise" if synthetic else args.wav
    print(f"\ndevice: {mx.device_info()['device_name']}")
    print(f"model:  whisper encoder, {dims.n_audio_layer} layers, width {dims.n_audio_state}")
    print(f"input:  {src}, mel {tuple(mel.shape)}")
    print(f"encoder weights: {w:.1f} MB   recommended working set: {ws:.0f} MB")
    print(f"instrumented calls: {len(records)}")

    print("\nby module kind (leaves only)")
    print_by_kind(records)

    print(f"\nmemory, first {args.rows} leaf calls")
    print_memory_timeline(records[: args.rows], w, ws)

    print("\nreconciliation")
    print(f"  unprofiled encoder      {clean:8.2f} ms   <- the truth")
    print(f"  profiled encoder        {dirty:8.2f} ms   <- cost of measuring")
    print(f"  sum of leaf layers      {leaves:8.2f} ms   <- what a naive profiler reports")
    print(f"  observer overhead       {dirty - clean:8.2f} ms  ({dirty / clean:.2f}x)")
    print(f"  leaf sum vs truth       {leaves / clean * 100:8.1f}%")
    peak = mx.get_peak_memory() / 1024 / 1024
    print(f"\n  peak active {peak:.1f} MB ({peak / ws * 100:.1f}% of working set)")


if __name__ == "__main__":
    main()
