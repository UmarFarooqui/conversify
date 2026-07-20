# Running Conversify (local VLM voice agent)

Steps to get `python -m conversify.main dev` to a working session:
Whisper STT → Qwen3-VL (Ollama) → Kokoro TTS, tested via LiveKit Cloud Console.

Verified on: WSL2 Ubuntu 22.04, Python 3.10, NVIDIA RTX 2050 (4 GB), CUDA 12.2.

Two VLM backends are documented. **Qwen3-VL-2B on Ollama (GPU)** is the default
and the only one fast enough for real-time voice. **RynnBrain-2B on llama.cpp**
is an embodied-AI alternative that currently runs CPU-only — see
[Alternative backend: RynnBrain-2B](#alternative-backend-rynnbrain-2b).

---

## 0. Prerequisites (one time)

```bash
sudo apt install libsndfile1
```

**Ollama must be pinned to 0.22.1.** Version 0.30.x has a GPU-discovery
regression that silently falls back to CPU under WSL (`library=cpu`,
`total_vram=0B`).

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.22.1 sh
ollama --version   # must print 0.22.1
```

Never re-run the install script without `OLLAMA_VERSION=0.22.1`.

Optional but recommended — GPU memory tuning:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=true"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Never run `ollama` commands with `sudo` — it writes root-owned blobs to
`/root/.ollama` and causes read-only filesystem errors later.

---

## 1. Python environment (one time)

```bash
cd /mnt/c/proseries/git_repos/conversify
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is incomplete. Also install:

```bash
pip install soundfile librosa faster-whisper
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "livekit-agents[images]" pillow
pip install livekit-plugins-noise-cancellation
pip install livekit-plugins-turn-detector
pip install livekit-plugins-silero
```

`livekit-agents[images]` (Pillow) is required for vision. Without it, camera
frames fail to encode and the error surfaces misleadingly as
`Connection error` with `ImportError: You haven't included the 'images'
optional dependencies` buried in the traceback.

Then download model assets:

```bash
python -m conversify.main download-files
```

### Do not upgrade huggingface_hub in this venv

`pip install -U "huggingface_hub[cli]"` pulls hub 1.x **and** transformers 5.x,
which breaks faster-whisper and the turn detector with
`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`.
Known-good pins:

```bash
pip install "huggingface_hub==0.33.0" "hf-xet==1.1.4" "transformers==4.52.4"
python -c "import transformers, tokenizers, faster_whisper; print('ok')"
```

Download GGUFs with `curl` rather than the HF CLI to avoid this entirely.

---

## 2. Pull the VLM (one time)

```bash
ollama pull qwen3-vl:2b-instruct
```

The 2B (1.9 GB) fits the 4 GB card with headroom. The 4B spills to CPU
(33%/67%) and drops to ~0.2 tok/s — slow enough that voice turns time out and
look like a crash.

Verify GPU placement:

```bash
ollama run qwen3-vl:2b-instruct "hi"
ollama ps    # PROCESSOR should be ~100% GPU
```

---

## 3. Config check

`config.yaml` should have both blocks pointing at the 2B:

```yaml
llm:
  base_url: "http://localhost:11434/v1"
  model: "qwen3-vl:2b-instruct"
  max_tokens: 100          # keep answers short; TTS time scales with length

vision:
  use: true
  model: "qwen3-vl:2b-instruct"

tts:
  base_url: "http://localhost:8880/v1"
  voice: "af_bella"        # must exist in Kokoro's voice list
```

---

## Every launch

### 4. Start Ollama

```bash
sudo systemctl start ollama
curl http://localhost:11434/api/tags     # should return JSON
```

### 5. Start Kokoro TTS

The container does **not** survive a WSL or Docker restart. This is the most
common cause of "no greeting" — the agent starts fine and fails only when it
tries to speak.

```bash
docker start kokoro-tts
sleep 5
curl http://localhost:8880/v1/audio/voices     # must list af_bella
```

If that curl refuses the connection, nothing downstream will produce audio.

### 6. Launch the agent

```bash
cd /mnt/c/proseries/git_repos/conversify
source venv/bin/activate
python -m conversify.main dev
```

Forgetting `source venv/bin/activate` gives `ModuleNotFoundError: openai`.

Wait for the worker to register — first start takes ~2 min because the
inference executor warms up over the Windows filesystem.

### 7. Connect from LiveKit Cloud Console

1. Open the project console → Agents → Console
2. **End session** on any stale room first
3. **Start a session**
4. Allow **microphone**; click the camera icon and allow **camera** for vision
5. Wait for the spoken greeting

The agent auto-dispatches — the "No agent selected" dropdown is irrelevant.
Success looks like a second participant with `KIND=AGENT`.

---

## Verifying it works

| Test | Expected |
|---|---|
| Greeting | Audible within ~15 s of joining |
| "What is the capital of France?" | Spoken answer, TTFT ~2 s |
| "Describe the scene" | Accurate description of the live camera frame |
| Repeat scene question | `Pruned images from N older message(s)` in log; prompt tokens stay flat (~1.4k) |

---

## Alternative backend: RynnBrain-2B

[RynnBrain-2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B) is a
Qwen3-VL-2B-Instruct fine-tune from DAMO Academy aimed at embodied AI —
egocentric understanding, spatial grounding, affordances, trajectories, grasp
poses. Apache-2.0, same backbone and size as the stock 2B.

It **cannot** run under Ollama on this setup, and on CPU it is roughly 30x
slower per turn than Qwen3-VL on GPU. Use it for spatial/grounding evaluation,
not for live conversation.

### R1. Download the GGUF (one time)

Use the **static** repo — it ships the `mmproj` (vision projector). The
`-i1-GGUF` imatrix repo has language weights only, which silently gives you a
text-only model.

```bash
mkdir -p ~/rynn && cd ~/rynn
curl -L -o RynnBrain-2B.Q4_K_M.gguf \
  https://huggingface.co/mradermacher/RynnBrain-2B-GGUF/resolve/main/RynnBrain-2B.Q4_K_M.gguf
curl -L -o RynnBrain-2B.mmproj-f16.gguf \
  https://huggingface.co/mradermacher/RynnBrain-2B-GGUF/resolve/main/RynnBrain-2B.mmproj-f16.gguf
ls -lh    # expect ~1.2 GB and ~781 MB
```

### R2. Install llama.cpp (one time)

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Installs a single `llama` binary with subcommands. There is no `llama-server`
executable — `llama serve` is the equivalent.

**Why not Ollama:** `ollama create` from these GGUFs reports success, but the
runner segfaults on first inference (`llama runner terminated, exit status 2`),
with or without the mmproj, and with either a two-`FROM` Modelfile or the
language weights alone. Ollama 0.22.1's bundled llama.cpp predates these
`qwen3vl` conversions. Also avoid the `ollama run hf.co/...` shortcut — it pulls
the quant without the projector, so vision silently does nothing.

### R3. Start the server (every launch)

```bash
llama serve \
  --model ~/rynn/RynnBrain-2B.Q4_K_M.gguf \
  --mmproj ~/rynn/RynnBrain-2B.mmproj-f16.gguf \
  --n-gpu-layers 99 --ctx-size 4096 \
  --host 127.0.0.1 --port 8081
```

Leave it running in its own terminal. Confirm vision loaded:

```bash
curl http://localhost:8081/v1/models
```

Look for `"capabilities":["completion","multimodal"]`. If `multimodal` is absent,
the mmproj did not load and image queries will fail.

Expected startup warnings:

- `no usable GPU found, --gpu-layers option will be ignored` — the installer
  probes CUDA but currently serves a CPU build; this is the performance problem
- `Qwen-VL models require at minimum 1024 image tokens` — add
  `--image-min-tokens 1024` if grounding accuracy matters
- `control-looking token: 128247 '</s>' was not control-type` — harmless
- `rocm-probe: libhipblas.so.3: cannot open shared object file` — AMD probe on an
  NVIDIA box

### R4. Point conversify at it

In `config.yaml`, change **only** the endpoint and model keys. Keep every other
key — `parallel_tool_calls`, `tool_choice`, and `video_frame_interval` are read
with bracket access (`llm.py:60-61`, `main.py:153`) and their absence is a
`KeyError` at startup.

```yaml
llm:
  base_url: "http://localhost:8081/v1"
  model: "/home/uafarooq/rynn/RynnBrain-2B.Q4_K_M.gguf"
  api_key: "llamacpp"
  temperature: 0.7
  max_tokens: 200
  parallel_tool_calls: false
  tool_choice: "auto"

vision:
  use: true
  model: "/home/uafarooq/rynn/RynnBrain-2B.Q4_K_M.gguf"
  base_url: "http://localhost:8081/v1"
  api_key: "llamacpp"
  video_frame_interval: 2.0
```

The model id **is** the absolute path to the GGUF — that is what `/v1/models`
reports. `api_key` is ignored by llama.cpp but must be non-empty for the OpenAI
client.

Back up the working config first, so reverting is one command:

```bash
cp config.yaml config.yaml.qwen.bak
python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['llm']['tool_choice'], c['vision']['video_frame_interval'])"
```

Then start Kokoro and the agent exactly as in steps 5–7 above.

### R5. Verify the model directly

Test before involving the agent — a minute-long turn is hard to debug through
the Console:

```bash
python3 - <<'EOF'
import base64, json
b64 = base64.b64encode(open("/mnt/c/proseries/ara_sdk/test.jpeg","rb").read()).decode()
json.dump({"model":"/home/uafarooq/rynn/RynnBrain-2B.Q4_K_M.gguf",
 "max_tokens":200,
 "messages":[{"role":"user","content":[
  {"type":"text","text":"Describe this image."},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}}]}]},
  open("/tmp/req.json","w"))
EOF
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" --data-binary @/tmp/req.json
```

An accurate description confirms the mmproj path works end to end.

### Performance

Measured on this hardware, same camera, same questions:

| | Qwen3-VL-2B (Ollama, GPU) | RynnBrain-2B (llama.cpp, CPU) |
|---|---|---|
| Placement | 100% GPU, 1.9 GB | CPU only, 1.2 GB |
| Time to first token | ~1.8 s | ~64 s |
| End-to-end latency | ~10 s | ~73 s |
| Image prompt eval | included above | 42–61 s / ~1000–1900 tokens (22–31 t/s) |
| Generation | 22–39 t/s | 11–20 t/s |

Prompt eval dominates. Each new camera frame invalidates the KV cache — the
server logs `sim_best ≈ 0.32`, so almost nothing is reused between turns and the
full image is re-prefilled every time. Generation speed is acceptable; the image
is the cost.

**The fix is a CUDA-enabled llama.cpp**, either a prebuilt CUDA variant or a
source build with `-DGGML_CUDA=ON`. Until then RynnBrain is not viable for live
voice.

Two further notes from testing. Scene descriptions are comparable in quality to
stock Qwen3-VL-2B — the fine-tuning advantage shows up on grounding tasks, not
casual description, and those need explicit prompting (bounding boxes,
coordinates) plus `--image-min-tokens 1024`. And with `video_frame_interval: 2.0`
against a ~60 s inference, the stored frame is replaced many times mid-turn, so
answers may describe a scene several seconds stale; raising the interval to
10–15 s reduces the churn on CPU.

### Reverting to Qwen3-VL

```bash
cd /mnt/c/proseries/git_repos/conversify
cp config.yaml.qwen.bak config.yaml
ollama pull qwen3-vl:2b-instruct    # if it was removed
ollama ps                            # PROCESSOR should read ~100% GPU
```

Both backends can run side by side — Ollama on :11434 for the live loop,
`llama serve` on :8081 for RynnBrain queries where latency is acceptable.

---

## Troubleshooting

**No greeting, `httpx.ConnectError` to :8880** — Kokoro is down. Step 5.

**Agent answers "I can't see images"** — Pillow missing, or the frame never
reached the model. Confirm Ollama's vision path independently:

```bash
python3 - <<'EOF'
import base64, json
b64 = base64.b64encode(open("/path/to/test.jpeg","rb").read()).decode()
json.dump({"model":"qwen3-vl:2b-instruct","messages":[{"role":"user","content":[
  {"type":"text","text":"Describe this image."},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}}]}]},
  open("/tmp/req.json","w"))
EOF
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" --data-binary @/tmp/req.json
```

Build the request in a file — inlining base64 on the command line hits
`Argument list too long`.

**`ImportError: cannot import name 'noise_cancellation'`** — install the
matching `livekit-plugins-*` package. Same pattern for `turn_detector`,
`silero`.

**Turns get slower each exchange** — image pruning isn't active. `llm_node`
should call `_prune_old_images(chat_ctx)` after `process_image(chat_ctx)`.

**`inference is slower than realtime`, turn never completes** — LLM too large
for the GPU. Check `ollama ps` for CPU spillover.

**Ollama on CPU after an update** — 0.30.x regression. Reinstall 0.22.1 (step 0).

**`ollama create` succeeds but inference returns 500** — expected for the
RynnBrain GGUFs on Ollama 0.22.1. Use `llama serve` (R2).

**`llama-server: command not found`** — the current installer ships one binary;
use `llama serve`.

**`KeyError: 'tool_choice'` / `'video_frame_interval'` at startup** — keys were
dropped while editing `config.yaml`. Both are read with bracket access; restore
them (R4).

**`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`** —
version skew from an HF CLI upgrade. Reapply the pins in section 1.

### Harmless noise

- `Failed to warm up STT engine: ... Format not recognised` — warmup bug, real
  transcription works
- `cloud turn detector failed, falling back to local mini model`
- `No video track found yet, waiting...` — only means no camera published
- Deprecation warnings from silero / turn_detector / RoomInputOptions

---

## Known performance limits

Running from `/mnt/c/...` (Windows filesystem) makes startup and file I/O
slow. Moving the repo to the native WSL filesystem (`~/conversify`, with a
fresh venv) cuts cold-start time substantially.

Every turn attaches a camera frame when `vision.use: true`, including
text-only questions — roughly 950 extra tokens each. A keyword gate
(see/look/scene/camera) in `process_image` would avoid that.
