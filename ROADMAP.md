# mlxprof

`llama-bench` plus a roofline plus per layer detail, for macs, where none of those
three exist.

**This file is the plan.** If it is not written here it is not the plan. Update it
when a decision changes, do not leave decisions in chat.

---

## the problem

this is everything mlx tells you today about a running model:

```
Prompt: 35 tokens, 37.497 tokens-per-sec
Generation: 60 tokens, 296.972 tokens-per-sec
Peak memory: 0.338 GB
```

is 297 tokens/sec good? is there room to improve? if you optimise, what do you
change? none of those are answerable from three numbers.

between `mlx_lm.benchmark` at the coarse end and a `.gputrace` in xcode at the fine
end there is nothing. that gap is the product.

## what we say instead

measured on an M5 Pro, Qwen2.5-0.5B-Instruct-4bit:

```
model weights          265.1 MB
measured ceiling       231.7 GB/s

decode  297 tok/s x 265 MB = 76.9 GB/s -> 33.2% MBU   memory bound
prefill  37 tok/s                       -> compute bound, MBU is the wrong metric

headroom: 67% of your 232 GB/s is idle during decode
```

mlx says 297 tokens/sec. we say 297 is a third of what the machine can do, you are
memory bound, and two thirds of your bandwidth is idle. one of those tells you what
to do next.

---

## the ux

### the front door: two lines, then look in the browser

```python
import mlxprof
mlxprof.serve()                 # localhost:7878, opens your browser

# your normal mlx code, completely unchanged below this line
model, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
generate(model, tok, "hello", max_tokens=100)
```

no config, no flags, no artifact juggling. every forward pass and every generation
appears live in the browser as a run.

the friction of `--out run.json` then `mlxprof view run.json` is what stops people
using a tool. neatlogs proved the one-line-then-look-in-browser ergonomic is what
gets adopted, and we copy it.

### screen 1: the run list, newest first

```
  #  model                        bound     util    tok/s    peak
  3  Qwen2.5-0.5B-4bit            memory     33%    297.0    338M
  2  Qwen2.5-0.5B-4bit            memory     31%    281.4    338M
  1  whisper-large-v3-turbo       compute     --    248ms     2.2G
```

runs accumulate, so comparing two is selecting both. that is `diff` without a cli.

### screen 2: click a run, verdict first

```
  ┌───────────────────────────────────────────────┐
  │  MEMORY BOUND · 33% MBU · 67% headroom        │   <- the answer, top of page
  │  297 tok/s of a possible ~890                 │
  └───────────────────────────────────────────────┘

  where the time goes          where the memory goes
  Linear     92.1% ##########  weights  265M ########
  LayerNorm   5.2% #           kv        12M #
  other       2.7%             peak     338M

  accuracy: per layer inflated ~29%, totals exact
```

the verdict is the top of the page, not buried. the detail is underneath for when
you believe the verdict and want to act on it.

### the terminal, for people who will not edit their code

```
mlxprof run -- python my_script.py
mlxprof bench --model <repo>
```

`bench` prints one screen:

```
machine    Apple M5 Pro, 24 GB unified, 232 GB/s measured
model      265 MB weights, 4bit

            tok/s    bound      utilization    verdict
prefill      37.5    compute    -- % MFU
decode      297.0    memory     33% MBU        67% headroom

where decode time goes
  Linear        92.1%   ####################################
  LayerNorm      5.2%   ##
  other          2.7%   #

memory     weights 265 MB   kv 12 MB   peak 338 MB   1.8% of working set
accuracy   per layer numbers inflated ~29%, totals are exact
```

### the artifact, for sharing and comparing

```
mlxprof bench --model <repo> --out run.json
mlxprof diff bf16.json 4bit.json
```

```
metric              bf16      4bit      change
end to end        257.5ms   161.2ms     -37.4%
weights            1211MB    340MB      -71.9%
peak active        1971MB    892MB      -54.7%

slower layers
  blocks.7.mlp2     1.83ms    2.41ms     +31.7%   <- dequant overhead
faster layers
  blocks.0.attn     4.12ms    2.02ms     -51.0%
```

"i changed quantisation, what got faster and what got slower" is the actual loop of
optimisation work and neither torch.profiler nor perfetto makes it easy.

---

## todo

### done

- [x] per layer timing via per instance class swap, with honest reconciliation
- [x] memory attribution over a forward pass, weights / activations / buffer cache
- [x] decode loop profiling with kv cache growth, self vs cross
- [x] `tree_bytes` for sizing any nested array structure
- [x] `walkthrough.py`, six runnable steps explaining the mechanism
- [x] roofline: measured machine ceilings, MBU / MFU, names the binding resource
- [x] installable package with a `mlxprof` command, own venv, no voice dependency
- [x] `mlxprof bench` and `mlxprof ceiling`
- [x] real chip topology from sysctl, and a warning when the machine is too busy
- [x] `mlxprof.serve()`: whole process auto attach and a localhost viewer

### next, in order

- [ ] 7. model matrix: decoder-only llm at two quantisations, not just whisper
- [ ] 8. self time vs total in the tree (gprof convention, expected everywhere)
- [ ] 9. one schedule object for warmup (torch.profiler convention, kills a bug class)
- [ ] 10. sampling mode, undistorted ground truth to calibrate the instrumented
      numbers against (py-spy convention)
- [ ] 11. `mlxprof diff`

4 to 6 is the product. everything else is polish underneath it.

### later, voice layer

built on the core, and the only part that gets a real timeline, because voice is the
only case where tracks genuinely overlap.

- [ ] turn timeline: mic, vad, asr, llm, tts on one wall clock axis
- [ ] audio and mels attached to spans via a generic artifact hook
- [ ] latency budget assertions, ttfa, barge in, endpointing

---

## open problems

**the observer effect.** forcing eval to measure costs 1.3x on real models and 1.9x
on toys, so absolute per layer numbers inflate. a fixed calibration constant will
not work because the distortion scales with layer size. the honest fix is a sampling
mode (item 10) to get undistorted totals to reconcile against.

---

## solved

**auto attach with no model handed to us.** solved. wrap every class that defines its
own `__call__`, walking `nn.Module.__subclasses__()` for classes that already exist,
plus a `__init_subclass__` hook for classes defined later. mlx_lm builds its model
classes at `load()` time, so the hook is what catches `TransformerBlock`, `Attention`
and friends. measured: 66 classes wrapped at install, 76 after loading a qwen model.

## decided, do not relitigate

- **not a timeline.** a forward pass is strictly sequential, so a gantt chart is a
  bar chart with a wasted axis. timelines earn their place only for the generation
  loop over steps, and for voice where tracks genuinely overlap.
- **not a monitor.** neatlogs and grafana watch fleets in production. nobody serves
  traffic from 500 macs. this is a development tool, the server is a viewer for
  runs, not a always-on dashboard.
- **general core, voice on top.** the test for which layer something belongs to:
  does it need to know about time inside a conversation? no means core, yes means
  voice. do not build a general ui.
- **we print our own error bars.** every profiler distorts. we are the only one that
  reports the distortion next to the result.
- **serve defaults to run level, not per layer.** forcing eval at every boundary to get
  per layer times inflates the run. so `serve()` records an honest run total with one
  sync at the end and records no per layer times at all, and `detail=True` opts into the
  distortion. the run says which mode produced it.
- **verdict before detail.** the binding resource and the utilization go at the top
  of every screen. per layer tables are for after you believe the verdict.

---

## what we learned building it

- mlx is lazy, so a naive timer measures graph construction, ~200x off
- `nn.Module` has no hooks, and patching an instance is silently ignored because
  python looks up dunders on the type. per instance class swap is the way in
- forcing eval to measure costs 1.3x on real models, 1.9x on toys. bigger layers
  amortise the sync, so the tool is more honest on real models than on demos
- whisper turbo has 4 decoder layers, not 32. its cross attention kv is 58 MB and
  fixed, while self attention kv is 1 MB and growing. every decoder-only intuition
  about kv cache inverts for encoder-decoder models
- freed activations go to mlx's buffer pool, not back to the os, which is why memory
  does not drop when you free things
- first decode step is 11x slower than warm, kernel compilation. any benchmark
  without warmup silently attributes that to the model
- `prompt_tps` on a cold `stream_generate` is ~10x low (231 vs 2400 tok/s). this bug
  bit us three times, in walkthrough step 3, in the whisper encoder, and in the first
  version of `bench`, which is why item 9 exists
- the memory cache in front of DRAM is shared between CPU and GPU (Apple Silicon CPU
  Optimization Guide 5.0), so a measured bandwidth ceiling is best case only. on a
  loaded machine the same model measured 163 tok/s against 453 on a quiet one, a 2.8x
  swing with no code change. the tool now reports load and says so
- `mx.device_info()` returns one string that hides real heterogeneity. an M5 Pro is
  5 Super + 10 Performance cores whose L1D differs 2x. read `hw.perflevel{N}.*` via
  sysctl, never the flat `hw.l1dcachesize`, which reports the weakest cores
- a ceiling should be measured best of N, not mean. averaging folds contention into
  the number you are calling the hardware limit. this cut ceiling spread from 35% to 6%
- the observer effect scales with how much real work a layer does. at seq 1 the layer
  split inflates 1687%, at 128 it is 291%, at 512 it is 165%. per layer profiling of a
  single decode step measures our own instrumentation, not the model

## references we took from

- roofline model and MBU / MFU: the binding resource framing, and the rule to report
  the utilization whose ceiling is 100% for that resource
- gprof: self time vs total time
- torch.profiler: `schedule(wait, warmup, active, repeat)` as a first class object
- py-spy: sampling as an alternative to instrumentation
- neatlogs: one line init, and readable hierarchy with key numbers as inline chips
- llama.cpp `llama-bench`: the local inference benchmark table shape
