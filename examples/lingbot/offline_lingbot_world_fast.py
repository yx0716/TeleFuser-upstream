"""Offline (non-streaming) driver for LingBot-World-Fast.

Streaming counterpart: ``stream_lingbot_world_fast.py`` (WebRTC service). This script
bypasses WebRTC / async / the browser entirely: it builds the pipeline once, runs the
chunked denoising loop to completion, and writes a single ``.mp4``. Much easier to
verify than the streaming path — run it, get a file.

The chunk loop mirrors ``LingBotWorldFastService._worker_loop`` minus the transport:

    runtime = pipeline.create_runtime(session_config)   # one-time warmup
    while runtime.active:                                # per-chunk denoise
        frames += pipeline.generate_next_chunk(runtime)  # control auto-applied from runtime
    save_video(frames, out_path, fps)

Camera control is optional: pass ``--action-path`` (a directory with ``poses.npy`` /
``intrinsics.npy``, optionally ``action.npy`` for ``--control-mode act``) to drive the
camera; omit it for a free-running (uncontrolled) clip.

Unlike the streaming example, text encoder + VAE default to CUDA here: an offline one-shot
has the GPU to itself, and keeping them on GPU avoids the slow CPU warmup the streaming
path pays (the text encoder + VAE condition-video encode dominate warmup on CPU).
Override with ``--device-text`` / ``--device-vae`` on tight-memory boxes.

Usage:
    TF_MODEL_ZOO_PATH=/path/to/model_zoo \\
        python examples/lingbot/offline_lingbot_world_fast.py \\
        --image /path/to/model_zoo/lingbot-world-fast/assets/teaser.png \\
        --prompt "a sunny street, the camera slowly moves forward" \\
        --save-path output/lingbot_offline.mp4
"""

from __future__ import annotations

import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig
from telefuser.utils.logging import logger
from telefuser.utils.video import save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="lingbot_world_fast_offline",
    # LingBot-World-Fast reuses the Wan2.2 base weights (VAE + T5 text encoder + ``google/umt5-xxl``
    # tokenizer) from the shared Wan2.2-I2V-A14B directory; the DiT fast weights live in their own
    # ``lingbot-world-fast`` directory, given as an absolute path so the pipeline keeps it standalone.
    checkpoint_dir=TF_MODEL_ZOO_PATH + "/Wan2.2-I2V-A14B",
    fast_checkpoint_subdir=TF_MODEL_ZOO_PATH + "/lingbot-world-fast",
    control_type="cam",
    # Offline default: keep everything on GPU for fast warmup (one-shot, GPU is free).
    dit_device="cuda",
    vae_device="cuda",
    text_device="cuda",
    device_id=0,
    max_area=480 * 832,
    torch_dtype=torch.bfloat16,
)


def build_pipeline(device_text: str, device_vae: str) -> LingBotWorldFastPipeline:
    dtype = PPL_CONFIG["torch_dtype"]
    # init() loads VAE / text encoder / DiT itself from the config paths and ignores the manager,
    # so an empty ModuleManager is just a signature-satisfying placeholder here.
    mm = ModuleManager(device="cpu")
    pipeline = LingBotWorldFastPipeline(device=PPL_CONFIG["dit_device"], torch_dtype=dtype)
    pipeline.init(
        mm,
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=PPL_CONFIG["checkpoint_dir"],
            fast_checkpoint_subdir=PPL_CONFIG["fast_checkpoint_subdir"],
            vae_config=ModelRuntimeConfig(
                device_type=device_vae,
                device_id=PPL_CONFIG["device_id"],
                torch_dtype=dtype,
            ),
            text_encoding_config=ModelRuntimeConfig(
                device_type=device_text,
                device_id=PPL_CONFIG["device_id"],
                torch_dtype=dtype,
            ),
            dit_torch_dtype=dtype,
            control_type=PPL_CONFIG["control_type"],
            max_area=PPL_CONFIG["max_area"],
        ),
    )
    return pipeline


def generate(pipeline: LingBotWorldFastPipeline, session_config: LingBotWorldFastSessionConfig) -> list[Image.Image]:
    """Run warmup + the full chunked denoising loop, returning all generated frames."""

    def on_progress(stage: str, **data: object) -> None:
        logger.info(f"[lingbot] {stage} {data if data else ''}")

    t0 = time.time()
    runtime = pipeline.create_runtime(session_config, progress_callback=on_progress)
    logger.info(
        f"runtime ready in {time.time() - t0:.1f}s: "
        f"{runtime.width}x{runtime.height}, {len(runtime.noise_chunks)} chunks"
    )

    frames: list[Image.Image] = []
    t_gen = time.time()
    # Mirror the service worker loop: pull chunks until the runtime stops yielding frames.
    while runtime.active and runtime.current_chunk_index < len(runtime.noise_chunks):
        chunk_frames = pipeline.generate_next_chunk(runtime, progress_callback=on_progress)
        if not chunk_frames:
            break
        frames.extend(chunk_frames)
    logger.info(f"generated {len(frames)} frames in {time.time() - t_gen:.1f}s")
    return frames


@click.command()
@click.option("--image", required=True, help="Input (first-frame) image path")
@click.option("--prompt", default="", help="Text prompt")
@click.option("--save-path", default="", help="Output mp4 path (default: ./<example>.mp4 or TELEAI_EXAMPLE_OUTPUT_DIR)")
@click.option("--action-path", default=None, help="Optional camera-control dir (poses.npy/intrinsics.npy[/action.npy])")
@click.option("--control-mode", default="cam", type=click.Choice(["cam", "act"]), help="Camera vs action control")
@click.option("--frame-num", default=81, type=int, help="Requested frame count")
@click.option("--chunk-size", default=3, type=int, help="Latent chunk size")
@click.option("--sample-shift", default=10.0, type=float, help="Sampler shift")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--fps", default=16, type=int, help="Output video FPS")
@click.option("--max-sequence-length", default=512, type=int, help="Max text sequence length")
@click.option("--device-text", default=PPL_CONFIG["text_device"], help="Text encoder device (cuda/cpu)")
@click.option("--device-vae", default=PPL_CONFIG["vae_device"], help="VAE device (cuda/cpu)")
def main(
    image: str,
    prompt: str,
    save_path: str,
    action_path: str | None,
    control_mode: str,
    frame_num: int,
    chunk_size: int,
    sample_shift: float,
    seed: int,
    fps: int,
    max_sequence_length: int,
    device_text: str,
    device_vae: str,
) -> None:
    """Offline image-to-video generation with LingBot-World-Fast."""
    pipeline = build_pipeline(device_text=device_text, device_vae=device_vae)

    session_config = LingBotWorldFastSessionConfig(
        prompt=prompt,
        image=Image.open(image).convert("RGB"),
        control_mode=control_mode,
        fps=fps,
        chunk_size=chunk_size,
        frame_num=frame_num,
        sample_shift=sample_shift,
        seed=seed,
        max_sequence_length=max_sequence_length,
        action_path=action_path,
    )

    frames = generate(pipeline, session_config)
    if not frames:
        raise RuntimeError("No frames generated")

    if not save_path:
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        save_path = os.path.join(output_dir, "offline_lingbot_world_fast.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_video(frames, save_path, fps=fps, quality=6)
    logger.info(f"Saved {len(frames)} frames to {save_path}")


if __name__ == "__main__":
    main()
