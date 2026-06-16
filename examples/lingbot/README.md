# LingBot-World-Fast examples

Two ways to drive the `LingBotWorldFastPipeline`:

| Script | Mode | Use when |
|--------|------|----------|
| `offline_lingbot_world_fast.py` | Offline batch → `.mp4` | You just want to generate / verify a clip. No browser, no network. |
| `stream_lingbot_world_fast.py` | Bidirectional WebRTC service | Interactive, real-time camera control from a browser. |

Both reuse the **Wan2.2 base weights** (VAE + T5 text encoder + `google/umt5-xxl` tokenizer)
from the shared `Wan2.2-I2V-A14B` directory, plus the DiT fast weights from `lingbot-world-fast`.
Set `TF_MODEL_ZOO_PATH` to your model-zoo root; the scripts build the full paths.

> Weight layout note: on some hosts `lingbot-world-base-cam/` aggregates the base weights via
> symlinks into `/storage/...`. Those symlinks are **broken on nodes where `/storage` is not
> mounted**, which is why the scripts point `checkpoint_dir` directly at `Wan2.2-I2V-A14B`
> (the real files) instead of `lingbot-world-base-cam`.

## Offline (recommended for verification)

```bash
TF_MODEL_ZOO_PATH=/pvcplatform/model_zoo \
    python examples/lingbot/offline_lingbot_world_fast.py \
    --image /pvcplatform/model_zoo/lingbot-world-fast/assets/teaser.png \
    --prompt "a sunny street, the camera slowly moves forward" \
    --save-path output/lingbot_offline.mp4
```

Optional camera control: `--action-path <dir>` where `<dir>` holds `poses.npy` /
`intrinsics.npy` (and `action.npy` for `--control-mode act`). Text encoder + VAE default to
CUDA for fast warmup; use `--device-text cpu --device-vae cpu` on tight-memory boxes.

## Streaming (WebRTC)

```bash
# 1. server
TF_MODEL_ZOO_PATH=/pvcplatform/model_zoo \
    telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py -p 8088 --skip-validation
# 2. browser client (open the printed URL, enter a prompt, Connect, then arrow keys / D-pad)
python examples/stream_server/webrtc_bidirectional_demo.py \
    --server-url http://localhost:8088 \
    --image-path /pvcplatform/model_zoo/lingbot-world-fast/assets/teaser.png
```

Pass the input image as a **single** `--image-path` argument (a bare directory, or a path split
by a line break, silently degrades to "no image").

---

## Known blocker: streaming warmup is CPU-bound and slow

Observed while verifying the WebRTC path end-to-end (headless `aiortc` client driving the
server). Captured here so the next person does not re-discover it.

### What works

- **WebRTC transport is fully functional.** ICE reaches `connected`, the `telefuser`
  DataChannel exchanges JSON control, and the video track delivers frames. The `/offer` body
  and the DataChannel control protocol are correct.

### What is actually wrong

- **The frames you first see are a placeholder, not generated output.** On session start the
  service emits one **preview frame** — the input image (e.g. `teaser.png`) rescaled — and
  `FrameGeneratorTrack` repeats it at the target FPS until real chunks arrive. The "LingBot
  collage + WASD HUD" picture is just the contents of `teaser.png`, not model output. This is
  why `nvidia-smi` shows **0% GPU util** during this window.

- **Real generation is gated behind a very slow, CPU-bound warmup.** The DataChannel `status`
  stages progress only as far as:

  ```
  preview → initializing_runtime → loading_controls → encoding_prompt
          → (~39 s later) prompt_encoded → preparing_image → encoding_condition_video
  ```

  `runtime_ready` / `generating_chunk` / `chunk_sent` are only reached **after `create_runtime`
  finishes**, which was observed at **~100 s** into the session. The dominant cost is prompt /
  condition encoding running on **CPU** (`status` reports `device=cpu`) — a direct consequence
  of `stream_lingbot_world_fast.py` placing the VAE and text encoder on `cpu` (to leave GPU
  memory for the DiT during long sessions).

### Implications

- A client that disconnects before ~100 s never sees real diffusion — only the repeated preview
  frame — and GPU util stays at 0%. The pipeline is **not broken**; the warmup is just long and
  front-loaded onto the CPU.
- **For verification, prefer the offline script** (`offline_lingbot_world_fast.py`): it keeps
  text/VAE on GPU by default, so warmup is far faster, and it produces a concrete `.mp4` to
  inspect.
- To speed up the *streaming* path, move the text encoder (and optionally VAE) to GPU in
  `stream_lingbot_world_fast.py`'s `PPL_CONFIG` — at the cost of GPU memory headroom.

### Viewing the stream from a local browser

The demo HTTP server (`8091`) and the stream server (`8099`/`8088`) both bind `0.0.0.0`. WebRTC
**media is UDP**, so a plain SSH `-L` tunnel forwards the page + signaling but not the video.
Two viable paths: (1) direct access if your machine can route to the box's intranet IP and its
UDP media ports; (2) the designed path — run a TURN server on the box and forward `8091`,
the signaling port, and TURN's `3478/tcp` over SSH so media relays over TCP (see
`--turn-url` in `webrtc_bidirectional_demo.py`).
