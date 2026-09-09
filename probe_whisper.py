import os
import sys
import time

VOICE = "/Users/mintu/coding/ml/voice/test-1"
os.environ.setdefault("HF_HOME", os.path.join(VOICE, "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_whisper import load_models
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim

from mlxprof import (
    LayerProfiler,
    leaf_total,
    mark_leaves,
    print_by_kind,
    print_memory_timeline,
    weight_mb,
    working_set_mb,
)

REPO = "mlx-community/whisper-large-v3-turbo"
WAV = os.path.join(VOICE, "kokoro_out.wav")


def main():
    audio, sr = sf.read(WAV)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    audio = audio.astype(np.float32)

    print(f"loading {REPO} ...")
    model = load_models.load_model(REPO)
    dims = model.dims
    mel = log_mel_spectrogram(pad_or_trim(mx.array(audio), 480000), n_mels=dims.n_mels)
    mel = mel[None]
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

    print(f"\ndevice: {mx.device_info()['device_name']}")
    print(f"model:  whisper encoder, {dims.n_audio_layer} layers, width {dims.n_audio_state}, mel {tuple(mel.shape)}")
    print(f"encoder weights: {w:.1f} MB   recommended working set: {ws:.0f} MB")
    print(f"modules instrumented: {len(records)} calls")

    print("\nby module kind (leaves only)")
    print_by_kind(records)

    print("\nmemory, first 12 leaf calls")
    print_memory_timeline(records[:40], w, ws)

    print("\nreconciliation")
    print(f"  unprofiled encoder      {clean:8.2f} ms   <- the truth")
    print(f"  profiled encoder        {dirty:8.2f} ms   <- cost of measuring")
    print(f"  sum of leaf layers      {leaves:8.2f} ms   <- what a naive profiler reports")
    print(f"  observer overhead       {dirty - clean:8.2f} ms  ({dirty/clean:.2f}x)")
    print(f"  leaf sum vs truth       {leaves/clean*100:8.1f}%")
    print(f"\n  peak active {mx.get_peak_memory()/1024/1024:.1f} MB ({mx.get_peak_memory()/1024/1024/ws*100:.1f}% of working set)")


if __name__ == "__main__":
    main()
