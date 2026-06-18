# Latent Cache (Cross-Request Approximate Cache for Diffusion)

Latent cache reuses **the intermediate latent from a previous inference**
when an incoming prompt is similar enough to a prompt already served, so the
first N denoising steps can be skipped. The implementation lives in
`telefuser/cache_mem/` and is plugged into the FastAPI service through
`telefuser/service/cache/`.

## Latent Cache vs. Feature Cache

The two solve different problems:

|                  | Feature cache (see `feature_cache.md`)      | Latent cache (this doc)                              |
| ---------------- | ------------------------------------------- | ---------------------------------------------------- |
| Granularity      | Within a single inference, across timesteps | Across requests                                      |
| Reuse key        | Step index                                  | Prompt embedding similarity                          |
| Acceleration     | Skip approximable blocks                    | Skip the first N denoising steps                     |
| Module           | `telefuser/feature_cache/`                  | `telefuser/cache_mem/`                               |
| Persistence      | None (request lifetime only)                | KV on disk / distributed store + vector DB + metadata |

Use feature cache to speed up *one* inference; use latent cache to speed up
the **next** inference whose prompt is similar to a cached one. The two can
be enabled at the same time without interfering.

---

## Base Interface

Latent cache exposes two layers of interfaces: `LatentCache` is the
pipeline / service-facing facade on top, and `BaseCacheStrategy` is the
abstract base class underneath that decides hit logic and save behavior.

```python
class BaseCacheStrategy(ABC):
    @abstractmethod
    async def lookup(self, prompt: str, task_type: str) -> CacheResult:
        """Query the cache and return whether a hit occurred and the cached latent state."""
        pass

    async def apply(self, result: CacheResult) -> CacheResult:
        """Post-process the lookup result (e.g. load the latent)."""
        return result

    @abstractmethod
    async def save(
        self,
        prompt: str,
        latent_states_dict: Dict[int, torch.Tensor],
        num_frames: int,
        task_type: str,
        saved_steps: List[int],
        embedding_video_frames: Optional[List[Any]] = None,
    ) -> None:
        """Write back the intermediate latent of this inference to the cache."""
        pass


class LatentCache:
    async def lookup(self, task_request: Any) -> CacheResult: ...
    async def save(
        self,
        task_request: Any,
        latent_states_dict: Dict[int, torch.Tensor],
        num_frames: int,
        final_step: int,
        saved_steps: List[int],
        embedding_video_frames: Optional[List[Any]] = None,
    ) -> None: ...
    def shutdown(self) -> None: ...
    def purge_by_prompt(self, prompt: str, collection: str = "whole") -> bool: ...
```

Key fields of `CacheResult`:

| Field            | Type             | Meaning                                              |
| ---------------- | ---------------- | ---------------------------------------------------- |
| `hit`            | bool             | Whether a hit occurred                               |
| `skip_step`      | int              | Step to restart denoising from on hit (>0 means skipping the first N steps) |
| `cache_type`    | str              | Hit cache type, e.g. `approximate_cache`             |
| `similarity`    | float            | Vector search / rerank score                         |
| `latent_state`  | Tensor \| None   | Cached latent tensor returned on hit                 |
| `cached_prompt` | str              | Original prompt of the hit entry                     |

### Use in Model Forward

The pipeline forwards the `latent_data` injected by the service layer down
to the denoise stage. The denoising loop uses `skip_step` to decide where
to start, and snapshots intermediate latents at the steps listed in
`saved_steps`:

```python
# In the denoise stage (see telefuser/pipelines/wan_video/moe_dit_denoising.py)
cached_latent, effective_start_step, saved_steps = parse_latent_data(
    latent_data,
    expected_shape=tuple(latents.shape),
    total_steps=total_steps,
)
if cached_latent is not None:
    latents = cached_latent.to(device=latents.device, dtype=latents.dtype)

saved_steps_set = frozenset(saved_steps)
latent_states_dict: dict[int, torch.Tensor] = {}

for progress_id, timestep in enumerate(timesteps[effective_start_step:]):
    absolute_step = effective_start_step + progress_id
    # snapshot BEFORE scheduler.step: step k stores the latent that enters step k
    if absolute_step in saved_steps_set:
        latent_states_dict[absolute_step] = latents.detach().cpu()
    noise_pred = self.predict_noise_with_cfg(...)
    latents = self.scheduler.step(noise_pred, timesteps[absolute_step], latents)

# pipeline returns the payload alongside the latent so the service layer
# can write it back asynchronously
latent_payload = {
    "latent_states_dict": latent_states_dict,
    "saved_steps": saved_steps,
    "final_step": total_steps - 1,
}
return latents, latent_payload
```

`parse_latent_data` (`telefuser/pipelines/wan_video/latent_data_utils.py`)
performs shape and range validation: if the shape mismatches or `skip_step`
is out of range, the cache is silently dropped and the pipeline falls back
to full denoising, so the main path is never poisoned by a bad cache entry.

---

## Factory Function

The production path no longer constructs `LatentCache` directly. Instead,
the Cacheseek TeleFuser adapter builds a `(CacheService, TeleFuserCacheAdapter)`
pair from CLI arguments and the `CACHE_CONFIG` declared in the pipeline file:

```python
from cacheseek.adapters.telefuser.cache_factory import CacheServiceFactory

cache_service, cache_adapter = CacheServiceFactory.create_cache_service(
    ppl_file="examples/wan_video/wan22_14b_text_to_video_service.py",
    enable_latent_cache=True,
    cache_mode="read_write",  # "read_write" / "read_only" / "write_only"
)
```

`create_cache_service` does the following internally:

1. Loads `CACHE_CONFIG` (a dict or a `CacheConfig` instance) from `ppl_file`
   as the base configuration.
2. Overrides the final config with the CLI's `enable_latent_cache` /
   `cache_mode`.
3. Initializes the cache log sink up front.
4. Builds the Cacheseek storage, vector store, metadata manager, and strategy.
5. Returns the framework-agnostic `CacheService` plus the TeleFuser adapter
   used for `build_query`, `apply_resume`, and `on_response`.

Manual construction should use Cacheseek primitives directly when needed:

```python
from pathlib import Path

from cacheseek.core.config import CacheConfig
from cacheseek.core.lifecycle import CacheService

config = CacheConfig(
    enable_latent_cache=True,
    latent_cache_dir="./latent_cache/wan22_t2v",
    cache_strategy_type="video_approximate",
    vector_dim=2048,
)
cache_service = CacheService.from_config(config)
```

The strategy class is selected by Cacheseek's TeleFuser factory from
`cache_strategy_type`. Custom strategies should be registered in Cacheseek,
not under `telefuser.cache_mem`.

```python
from cacheseek.core.strategies import get_strategy_class, register_strategy

register_strategy("video_approximate", VideoBasedApproximateCache)  # already registered by default
strategy_cls = get_strategy_class("video_approximate")
```

---

## VideoBasedApproximateCache

The only production strategy implementation is
`VideoBasedApproximateCache`, which combines:

- **Prompt encoding**: `Qwen3-VL-Embedding` encodes the prompt into a vector
  that is written to the vector store.
- **Video encoding**: during save, several frames of the generated video are
  encoded into the same vector space, used as the similarity basis for
  future hits.
- **Optional rerank**: when `rerank_enabled` is on, `Qwen3-VL-Reranker`
  performs cross-encoder reranking over the top-k candidates.
- **Shared backend**: when text and video embedding configs end up loading
  the same model on the same device, the two automatically share a single
  `Qwen3VLEncoder` instance, saving roughly 5 GB of GPU memory and one cold
  load.

### How VideoBasedApproximateCache Works

#### Write Path

When a request finishes, the pipeline returns `latent_payload` containing
the per-step latents plus video frames used for prompt similarity. The service
layer passes that payload through `TeleFuserCacheAdapter.on_response`, then
calls `CacheService.save(cache_query, outputs)`, which enqueues it onto the
Cacheseek async save worker:

1. Writes each step's latent to the KV store under a key shaped like
   `f"{cache_id}_step{step}"`.
2. Encodes the video frames with `Qwen3-VL-Embedding` and upserts the
   vector into the vector store (default collection name `video`).
3. Registers `cache_id -> {prompt, saved_steps, size_mb, ...}` in metadata,
   persisting `prompt_index.json` and `cache_meta.json`.

If any step fails, all the latents / vectors / metadata that were already
written are rolled back cleanly to avoid an inconsistent state.

#### Hit Path

When a new request arrives, the service layer runs
`adapter.build_query -> cache_service.lookup -> adapter.apply_resume`:

1. Waits on `vector_update_idle` to make sure the vector upsert from the
   previous async save has been committed.
2. Calls Cacheseek lookup: encodes the new prompt, queries the top-k
   approximate caches in the vector store, optionally reranks with
   Qwen3-VL-Reranker, and compares against the threshold to decide on a
   hit.
3. On a hit, loads the latent tensor for `skip_step` from the KV store and
   wraps it into a `CacheResult`.

The `latent_data` dict the pipeline receives includes `cached_latent`,
`skip_step`, and `saved_steps`. The pipeline restarts the denoise loop at
`skip_step` and snapshots this run's latents according to `saved_steps` —
that is how the cache keeps growing.

### Cache Parameters

The core parameters used by `VideoBasedApproximateCache`:

| Parameter                    | Type  | Description                                         |
| ---------------------------- | ----- | --------------------------------------------------- |
| `key_steps`                  | list  | Step list at which the pipeline is asked to snapshot |
| `video_similarity_threshold` | float | Lower bound for a vector-search hit                 |
| `rerank_enabled`             | bool  | Whether to rerank the top-k with Qwen3-VL-Reranker  |
| `rerank_top_k`               | int   | Number of candidates fed into rerank                |
| `rerank_score_threshold`     | float | Lower bound for a hit when rerank is enabled        |
| `video_embedding_max_frames` | int   | Max frames sampled when encoding video              |
| `video_vector_collection`    | str   | FAISS collection name (default `video`)             |

> Which step to restart from after a hit is decided by
> `_determine_skip_step`: in the current implementation, it skips to step 5
> when `similarity` is above the rerank threshold and `5 ∈ saved_steps`,
> otherwise it is treated as a miss. Override this method in a subclass to
> customize the skip policy.

### Using VideoBasedApproximateCache

Declaring `CACHE_CONFIG` in the pipeline file is enough to enable it
(`CacheServiceFactory` picks it up automatically at service startup):

```python
# examples/wan_video/wan22_14b_text_to_video_service.py
CACHE_CONFIG = dict(
    enable_latent_cache=True,
    latent_cache_dir=os.getenv("TELEFUSER_LATENT_CACHE_DIR", "./latent_cache/wan22_t2v"),
    cache_mode="write_only",
    kv_store_type="local_file",
    vector_store_type="faiss",
    # Qwen3-VL-Embedding-2B hidden_size=2048; must match the vector_store dim.
    vector_dim=2048,
    key_steps=[5, 10, 15, 20, 25],
    video_embedding_enabled=True,
    video_embedding_model_path=os.getenv("QWEN3VL_EMBEDDING_PATH", ""),
    video_embedding_max_frames=16,
    text_embedding_device_id=1,
    video_embedding_device_id=1,
    video_vector_collection="video",
    rerank_enabled=True,
    rerank_model_path=os.getenv("QWEN3VL_RERANKER_PATH", "/storage/model_zoo/Qwen3-VL-Reranker-2B"),
    rerank_device_id=int(os.getenv("TELEFUSER_RERANK_DEVICE_ID", "0")),
    rerank_top_k=5,
    rerank_score_threshold=0.85,
)
```

The pipeline file only needs to expose `run_with_file` plus a `CACHE_CONFIG`
dict for Cacheseek configuration:

- `run_with_file(pipeline, **task_data) -> dict`: feeds `latent_data` into
  the pipeline and returns `latent_payload` as part of the result so the
  service layer can write it back to the cache.

---

## Example Scripts

| Pipeline                          | Script                                                            | Notes                                |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------------ |
| Wan2.2 14B T2V (cache enabled)    | `examples/wan_video/wan22_14b_text_to_video_service.py`           | Full latent cache configuration example |
| Wan2.2 14B T2V (cache disabled)   | `examples/wan_video/wan22_14b_text_to_video_service_nocache.py`   | Same interface, cache off, control group |

Start the service:

```bash
telefuser serve examples/wan_video/wan22_14b_text_to_video_service.py \
    --port 8000 \
    --enable-latent-cache \
    --cache-mode read_write
```

---

## CacheConfig Field Reference

The full definition lives in `telefuser/cache_mem/config.py`; the tables
below show the defaults.

### Basic

| Field | Default | Description |
|---|---|---|
| `enable_latent_cache` | `False` | Master switch; toggled by CLI `--enable-latent-cache`. |
| `cache_mode` | `READ_WRITE` | One of `READ_WRITE` / `READ_ONLY` / `WRITE_ONLY`. |
| `latent_cache_dir` | `./latent_cache` | Root directory for storage, metadata, FAISS, and logs. |
| `max_cache_size_gb` | `10` | Soft eviction cap (LRU by `last_access_time`). |

### Logging

| Field | Default |
|---|---|
| `cache_log_enabled` | `True` |
| `cache_log_dir` | `{latent_cache_dir}/logs` |
| `cache_log_level` | `DEBUG` |
| `cache_log_rotation` | `100 MB` |
| `cache_log_retention` | `7 days` |

### KV / Vector Backend

| Field | Default | Description |
|---|---|---|
| `kv_store_type` | `local_file` | Or `fluxon` (stub). |
| `vector_store_type` | `faiss` | Or `qdrant` (stub). |
| `vector_dim` | `2048` | Must match the embedder output dim. |
| `faiss_index_dir` | `{latent_cache_dir}/faiss` | |
| `qdrant_url` / `qdrant_api_key` | `""` / `None` | Configure once Qdrant is wired up for real. |
| `cache_strategy_type` | `video_approximate` | Key in the strategy registry. |

### Strategy and Embedding

| Field | Default | Description |
|---|---|---|
| `key_steps` | `[0, 1, 2, 3, 4, 5]` | Step list at which the pipeline is asked to snapshot. |
| `lookup_mode` | `video` | |
| `video_embedding_enabled` | `True` | |
| `video_embedding_model_path` | `Qwen/Qwen3-VL-Embedding-2B` | |
| `video_embedding_max_frames` | `16` | |
| `video_embedding_fps` | `1.0` | |
| `text_embedding_model_path` | `""` | Empty means reuse the video embedder. |
| `video_similarity_threshold` | `0.10` | Lower bound for a vector-search hit. |
| `rerank_enabled` | `False` | When on, rerank the top-k with Qwen3-VL-Reranker. |
| `rerank_top_k` | `5` | |
| `rerank_score_threshold` | `0.90` | Lower bound for a hit when rerank is enabled. |

When the text and video embedding configurations end up loading the same
model on the same device, `VideoBasedApproximateCache` lets them share a
single `Qwen3VLEncoder` instance, saving roughly 5 GB of GPU memory and one
cold load.

### Async Save

| Field | Default | Description |
|---|---|---|
| `save_async_enabled` | `True` | Offload `save` onto the worker thread. |
| `save_queue_size` | `2` | `0` means unbounded. |
| `save_on_full` | `drop` | `drop` / `sync` / `downgrade` (downgrade is TODO). |
| `save_queue_warn_threshold` | `8` | Log a warning when queue depth exceeds this value. |
| `vector_wait_warn_s` | `2.0` | Log a warning when `lookup` waits on the vector barrier longer than this. |
| `vector_wait_timeout_s` | `120.0` | Give up the barrier after timeout and treat as miss. |
| `flush_on_shutdown` | `True` | `CacheService.shutdown` drains the queue first. |

### The Three Cache Modes

| Mode | Effect |
|---|---|
| `READ_WRITE` | Lookup hits, and writes are also persisted. The default. |
| `READ_ONLY` | Lookup hits, but the cache is not updated. Useful during canary rollouts. |
| `WRITE_ONLY` | Lookup always misses, only accumulating cache. Common when warming up a cache against a benchmark. |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                  LatentCache (facade)                  │
│                                                        │
│  ├─ Strategy        VideoBasedApproximateCache          │
│  │     ├─ prompt_encoder  Qwen3-VL-Embedding           │
│  │     ├─ video_encoder   Qwen3-VL-Embedding (shared)  │
│  │     └─ reranker        Qwen3-VL-Reranker (optional) │
│  │                                                     │
│  ├─ KVStore         LocalFileKVStore | FluxonKVStore*  │
│  ├─ VectorStore     FAISSVectorStore | QdrantStore*    │
│  └─ MetadataManager LocalCacheMetadataManager          │
└──────────▲─────────────────────────────────────────────┘
           │ via CacheService (async writeback wrapper)
           │
   FastAPI request thread / pipeline
```

The Fluxon / Qdrant backends marked with `*` are still stubs (they raise
`NotImplementedError`); the production path only goes through
`LocalFileKVStore` + `FAISSVectorStore`.

`CacheService` owns the background async writeback thread plus a barrier
called `vector_update_idle`, which prevents a `lookup` from reading a stale
index before the previous `save` finishes its vector upsert.
