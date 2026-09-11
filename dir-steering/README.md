# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized hidden-width direction per normal
transformer layer. During inference, ds4 can apply the edit after attention
outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

The file shape depends on the model:

- DeepSeek V4 Flash: `43 x 4096`.
- GLM 5.3 Flash: `45 x 4096`. The separate MTP predictor layer is omitted.
- Qwen 3.8-Flash-Next: `48 x 2560`.
- Qwen 3.5-2B: `24 x 2048`.

GLM 5.2 steering is not implemented.

On the Qwen family the edit lands on the sub-layer output, before the
hyper-connection combine spreads it into the wide residual: the same place in
the layer the other families steer.

## Runtime Options

```text
--dir-steering-file FILE   load one f32 direction per normal model layer
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

Over HTTP those scales are defaults, not constants: a request may override
either one for its own duration, which makes a running server a convenient
place to sweep a scale without restarting it.

```sh
"steering": {"ffn": 1.5, "attn": 0}
```

Members are independent and optional (an omitted member keeps the startup
value), and all four server endpoints accept the object. The override reaches
prefill as well as decode of that request, but it is deliberately not part of
the KV cache identity: tokens already in the cache keep the scales they were
computed with, so the observable effect is diluted in proportion to how much
of the context is cached, and it converges as new tokens accumulate. That is
the same contract as the interactive `/steer`, which is why a change is cheap
and why a response depends on the scales used earlier in the same cached
conversation.

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## GLM 5.3 Example

Build a GLM 5.3 direction from paired target and control prompt lists:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile glm-5.3-flash \
  --ds4 ./ds4 \
  --model gguf/GLM-5.3-Flash-Q2.gguf \
  --good-file /path/to/target-prompts.txt \
  --bad-file /path/to/control-prompts.txt \
  --out dir-steering/out/glm53-direction.json \
  --component ffn_out \
  --ctx 512
```

Generated `.f32` vectors are local artifacts and are not stored in the
repository. GLM 5.3 steering works with `--mtp`, `ds4-server`, native session
batching, and two-Mac tensor parallelism. For tensor parallelism, pass the same
steering file and scales to both the worker and coordinator.

## Verbosity Example

The bundled example builds a style direction from 100 paired prompts. Each pair
asks for the same information in two ways:

- `examples/succinct.txt`: terse target prompts.
- `examples/verbose.txt`: detailed contrast prompts.

Because the extracted direction is `succinct - verbose`, negative FFN scales
make answers shorter, while positive FFN scales tend to make answers longer and
more explanatory.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile deepseek-v4-flash \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Try a terse run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Try a verbose run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

The same vector can be used in either direction. The sign is the important part:

- negative scale amplifies the succinct target direction;
- positive scale suppresses that direction and usually gives the model more room
  to elaborate.

## Evaluating Scales

Use the sweep helper to test several strengths on a fixed prompt set:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Start with FFN scales between `-1` and `2`. If the model becomes repetitive,
ignores the prompt, or starts losing factual content, the scale is too strong.
For this example, `-1` is a good first terse setting and `2` is a good first
verbose setting. Strong negative scales such as `-2` or `-3` can over-amplify
the terse direction and collapse into repetition on some prompts.

## Observed Effect

With the 100-pair vector built from the commands above, local greedy checks
showed the expected behavior:

- Prompt: `Explain why databases use indexes.`
- `--dir-steering-ffn -1`: 67 words, one compact paragraph.
- `--dir-steering-ffn 0`: 136 words, structured explanation.
- `--dir-steering-ffn 1`: 140 words, structured explanation with more detail.

On a prompt that the unsteered model already answered briefly, positive steering
made the expansion more visible:

- Prompt: `What does DNS do?`
- `--dir-steering-ffn 0`: 44 words.
- `--dir-steering-ffn 2`: 171 words, with sections and step-by-step detail.

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures DS4 activations from the same local GPU graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file.

Concept removal:

1. Put concept-heavy prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a positive FFN scale.

Concept amplification:

1. Put desired concept prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a negative FFN scale.

Style control:

1. Put prompts for the target style in `good-file`.
2. Put contrasting style prompts in `bad-file`.
3. Use negative scale to amplify the target style, positive scale to reduce it.

The method is not a fine-tune. It is a low-rank runtime edit, so it works best
for coarse behavior, topic, or style directions that are consistently present in
the activation captures.

## Capturing Through a Running Server

Every capture above starts a fresh `ds4`, which reloads the whole checkpoint:
fine for a small model, but on a 70 GB one a single capture costs more than the
extraction itself, and 100 prompt pairs takes over an hour.  `--server` reads
the captures from a `ds4-server` that already holds the model instead, one
request per prompt:

```sh
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/dir-dump/d \
DS4_METAL_GRAPH_DUMP_NAME=ffn_out \
DS4_METAL_GRAPH_DUMP_POS=0 \
./ds4-server -m gguf/Qwen3.8-Flash-Next-NVFP4.gguf --ctx 262144 \
  --host 0.0.0.0 --port 8000 --kv-disk-dir /tmp/ds4-kv

python3 dir-steering/tools/build_direction.py \
  --profile qwen3.8-flash-next \
  --server http://127.0.0.1:8000 \
  --dump-prefix /tmp/dir-dump/d \
  --model-name qwen3.8-flash-next \
  --good-file targets.txt \
  --bad-file controls.txt \
  --out dir-steering/out/qwen-direction.json \
  --component ffn_out
```

The same 100 pairs take about a minute this way.

- `--dump-prefix` must be the exact `DS4_METAL_GRAPH_DUMP_PREFIX` the server was
  started with.  The builder reads `<prefix>_<component>-<layer>_pos0.bin` after
  each request, so the files are copied out before the next request overwrites
  them; run one extraction at a time.
- The dump environment is read once, when the hooks first run, so it has to be
  set on the server at start-up — and the server binary has to be the one built
  with these hooks.
- Requesting a dump disables the captured decode graphs, matching the guard the
  batched path already uses: a dump synchronizes and restarts the command
  batch, which a replay cannot express.
- The capture is the **last row** of the chunk, i.e. the prompt's last token.
  It has to be: row 0 is the prompt's first token, which a system prompt shared
  by every prompt makes identical, and a good/bad pair of identical captures
  normalizes to a zero direction that quietly does nothing.

### Tool-Calling Directions

A direction over tool-calling behaviour only exists if the model is offered the
tools while capturing: without them both sides of a pair fall back to "I can't
do that" and the captures come out identical.  The `ds4` CLI has no way to send
tools, so this works through `--server` only.

- `--tools-mock` offers the bundled `read` / `write` / `exec` trio
  (`examples/tools_mock.json`).
- `--tools-file FILE` offers your own array in OpenAI function format.

```sh
python3 dir-steering/tools/build_direction.py \
  --profile qwen3.8-flash-next \
  --server http://127.0.0.1:8000 --dump-prefix /tmp/dir-dump/d \
  --model-name qwen3.8-flash-next --tools-mock \
  --good-file reads.txt --bad-file execs.txt \
  --out dir-steering/out/tool-direction.json --component ffn_out
```

Check the split before extracting anything: send one prompt from each side and
compare the responses (`finish_reason` is `tool_calls` or not).  If both sides
emit a tool call, or both refuse, the pair carries no signal — the direction
will come out zero and apply as a no-op that reports no error.

The command above separates "read a file" from "run a command": a behaviour the
model exercises freely, so the machinery can be validated without any content
that is hard to check.
