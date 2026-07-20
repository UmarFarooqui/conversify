# Running Conversify (local VLM voice agent)

Steps to get `python -m conversify.main dev` to a working session:
Whisper STT → Qwen3-VL (Ollama) → Kokoro TTS, tested via LiveKit Cloud Console.

Verified on: WSL2 Ubuntu 22.04, Python 3.10, NVIDIA RTX 2050 (4 GB), CUDA 12.2.

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
