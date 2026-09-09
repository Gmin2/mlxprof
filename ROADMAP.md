# mlxprof

`llama-bench` plus a roofline plus per layer detail, for macs, where none of those
three exist.

## the problem

this is everything mlx tells you today about a running model:

```
Prompt: 35 tokens, 37.497 tokens-per-sec
Generation: 60 tokens, 296.972 tokens-per-sec
Peak memory: 0.338 GB
```

is 297 tokens/sec good? is there room to improve? if you optimise, what do you
change? none of those are answerable from three numbers.

between `mlx_lm.benchmark` at the coarse end and a `.gputrace` in xcode at the
fine end there is nothing. that gap is the product.

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

## the ux

one line in, browser out. no config, no flags, no artifact juggling.

```python
import mlxprof
mlxprof.serve()          # localhost:7878

# your normal mlx code, unchanged
model, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
generate(model, tok, "hello", max_tokens=100)
```

every forward pass and every generation shows up live in the browser as a run.
click a run and you get the verdict, the layer table, the memory curve.

for people who do not want to touch their code:

```
mlxprof run -- python my_script.py
mlxprof bench --model <repo>
```

and the artifact path, for sharing and comparing:

```
mlxprof bench --model <repo> --out run.json
mlxprof diff bf16.json 4bit.json
```

## what a run looks like

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

## plan

| # | piece | why | state |
|---|---|---|---|
| 1 | per layer timing, reconciled | the mechanism | done |
| 2 | memory attribution | where unified memory goes | done |
| 3 | decode loop + kv cache | generation, not forward passes | done |
| 4 | **roofline: MBU / MFU, name the binding resource** | tells you if optimising is even possible. the headline number | next |
| 5 | `mlxprof bench` cli | the front door, one command | |
| 6 | `mlxprof.serve()` + localhost viewer | the ergonomic that gets it used | |
| 7 | model matrix, not just whisper | decoder-only llm at 2 quantisations | |
| 8 | self time vs total | expected in every profiler (gprof) | |
| 9 | one schedule object | kills the warmup bug class (torch.profiler) | |
| 10 | sampling mode | undistorted ground truth to calibrate against (py-spy) | |
| 11 | `mlxprof diff` | the optimisation loop nobody supports well | |

## decided, do not relitigate

- **not a timeline.** a forward pass is strictly sequential, so a gantt chart is a
  bar chart with a wasted axis. timelines earn their place only for the generation
  loop over steps, and for voice where tracks genuinely overlap.
- **not a monitor.** neatlogs and grafana watch fleets in production. nobody serves
  traffic from 500 macs. this is a development tool, the server is a viewer.
- **general core, voice on top.** the test for which layer something belongs to: does
  it need to know about time inside a conversation? no means core, yes means voice.
  do not build a general ui.
- **we print our own error bars.** every profiler distorts. we are the only one that
  reports the distortion next to the result.

## what we learned building it

- mlx is lazy, so a naive timer measures graph construction, ~200x off
- `nn.Module` has no hooks, and patching an instance is silently ignored because
  python looks up dunders on the type. per instance class swap is the way in
- forcing eval to measure costs 1.3x on real models, 1.9x on toys. bigger layers
  amortise the sync, so the tool is more honest on real models than on demos
- whisper turbo has 4 decoder layers, not 32. its cross attention kv is 58 MB and
  fixed, while self attention kv is 1 MB and growing. every decoder-only intuition
  about kv cache inverts for encoder-decoder models
- freed activations go to mlx's buffer pool, not back to the os
