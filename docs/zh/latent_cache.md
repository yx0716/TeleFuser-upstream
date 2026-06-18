# Latent Cache（Diffusion 跨请求近似缓存）

Latent cache 用于在新到达的 prompt 和已经生成过的 prompt 足够相似
时**复用上一次推理的中间 latent**，跳过前若干步去噪。实现位于
`telefuser/cache_mem/`，通过 `telefuser/service/cache/` 接入 FastAPI 服务。

## Latent Cache 与 Feature Cache 的区别

两者解决的问题维度相异：

|      | Feature cache（参见 `feature_cache.md`） | Latent cache（本文档）       |
| ---- | ------------------------------------ | ----------------------- |
| 粒度   | 单次推理内、跨 timestep                     | 跨请求                     |
| 复用键  | step 索引                              | prompt embedding 相似度    |
| 加速目标 | 跳过可近似的 block                         | 跳过整次去噪的前 N 步         |
| 模块   | `telefuser/feature_cache/`           | `telefuser/cache_mem/`  |
| 持久化  | 无（只在请求生命周期内）                         | KV 磁盘/分布式存储 + 向量库 + 元数据 |

单次请求内推理加速用 feature cache；加速**多次**请求推理时
用 latent cache。两者可以同时启用、互不干扰。

---

## 基础接口

Latent cache 由两层接口构成：上层 `LatentCache` 是面向 pipeline / service
的外观类；下层 `BaseCacheStrategy` 是策略抽象基类，决定具体的命中判定与
保存逻辑。

```python
class BaseCacheStrategy(ABC):
    @abstractmethod
    async def lookup(self, prompt: str, task_type: str) -> CacheResult:
        """查询缓存，返回是否命中以及命中的 latent 状态。"""
        pass

    async def apply(self, result: CacheResult) -> CacheResult:
        """对 lookup 结果做后处理（例如加载 latent）。"""
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
        """将本次推理的中间 latent 写回缓存。"""
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

`CacheResult` 的关键字段：

| 字段            | 类型             | 含义                           |
| ------------- | -------------- | ---------------------------- |
| `hit`         | bool           | 是否命中                         |
| `skip_step`   | int            | 命中后从哪一步重新开始去噪（>0 表示跳过前 N 步）  |
| `cache_type`  | str            | 命中的缓存类型，如 `approximate_cache` |
| `similarity`  | float          | 向量检索 / rerank 的得分             |
| `latent_state`| Tensor \| None | 命中时返回的 latent 张量              |
| `cached_prompt` | str          | 命中条目原始 prompt               |

### 在模型 Forward 中的使用

Pipeline 将 service 层注入的 `latent_data` 传到 denoise stage，去噪循环根据
`skip_step` 决定从哪一步开始，同时按 `saved_steps` 对中间 latent 进行保存：

```python
# denoise stage（见 telefuser/pipelines/wan_video/moe_dit_denoising.py）
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
    # snapshot BEFORE scheduler.step：第 k 步存的是进入 step k 的 latent
    if absolute_step in saved_steps_set:
        latent_states_dict[absolute_step] = latents.detach().cpu()
    noise_pred = self.predict_noise_with_cfg(...)
    latents = self.scheduler.step(noise_pred, timesteps[absolute_step], latents)

# pipeline 在最后将 payload 一并返回，供 service 层异步写回缓存
latent_payload = {
    "latent_states_dict": latent_states_dict,
    "saved_steps": saved_steps,
    "final_step": total_steps - 1,
}
return latents, latent_payload
```

`parse_latent_data`（`telefuser/pipelines/wan_video/latent_data_utils.py`）会做
shape / 范围校验，shape 不一致或 `skip_step` 越界时会自动丢弃缓存并降级为
全量去噪，保证主链路不被污染。

---

## 工厂函数

线上路径不再直接构造 `LatentCache`，而是由 Cacheseek 的 TeleFuser 适配器根据
CLI 参数和 pipeline 文件中的 `CACHE_CONFIG` 生成 `(CacheService, TeleFuserCacheAdapter)`：

```python
from cacheseek.adapters.telefuser.cache_factory import CacheServiceFactory

cache_service, cache_adapter = CacheServiceFactory.create_cache_service(
    ppl_file="examples/wan_video/wan22_14b_text_to_video_service.py",
    enable_latent_cache=True,
    cache_mode="read_write",  # "read_write" / "read_only" / "write_only"
)
```

`create_cache_service` 内部会：

1. 从 `ppl_file` 加载 `CACHE_CONFIG`（dict 或 `CacheConfig` 实例）作为默认配置基础。
2. 用 CLI 的 `enable_latent_cache` / `cache_mode` 覆盖最终配置。
3. 提前初始化 cache 日志 sink。
4. 构造 Cacheseek 的存储、向量库、元数据管理器和策略。
5. 返回框架无关的 `CacheService`，以及用于 `build_query`、`apply_resume`
   和 `on_response` 的 TeleFuser 适配器。

需要直接构造时应使用 Cacheseek 原语：

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

策略类由 Cacheseek 的 TeleFuser factory 根据 `cache_strategy_type` 选择。
自定义策略应注册到 Cacheseek，而不是 `telefuser.cache_mem`：

```python
from cacheseek.core.strategies import get_strategy_class, register_strategy

register_strategy("video_approximate", VideoBasedApproximateCache)  # 默认已注册
strategy_cls = get_strategy_class("video_approximate")
```

---

## VideoBasedApproximateCache

线上唯一的策略实现是 `VideoBasedApproximateCache`，结合：

- **Prompt 编码**：`Qwen3-VL-Embedding` 把 prompt 编码成向量，写入向量检索库。
- **视频编码**：save 阶段对生成视频的若干帧编码至同一向量空间，作为命中时的相似度计算依据。
- **可选 rerank**：开启 `rerank_enabled` 后用 `Qwen3-VL-Reranker` 在 top-k 上
  做交叉编码精排。
- **共享后端**：当 text 和 video 的 embedding 配置最终落到同一个 model + device
  时，自动让两者共享同一个 `Qwen3VLEncoder` 实例，节约近 5 GB 显存和一次冷加载。

### VideoBasedApproximateCache 工作原理

#### 写入路径

请求结束后，pipeline 返回 `latent_payload`（含按步存储的 latent + 用于 prompt
相似度的视频帧）。服务层先经过 `TeleFuserCacheAdapter.on_response` 打包，
再调用 `CacheService.save(cache_query, outputs)`，由 Cacheseek 异步保存 worker
处理：

1. 将每个 step 的 latent 写到 KV，key 形如 `f"{cache_id}_step{step}"`。
2. 通过 `Qwen3-VL-Embedding` 将视频帧编码成向量，upsert 至 向量检索库（默认
   collection 名 `video`）。
3. 在 metadata 里登记 `cache_id -> {prompt, saved_steps, size_mb, ...}`，
   持久化 `prompt_index.json` 和 `cache_meta.json`。

任何一步失败，已写入的 latent / 向量 / metadata 都会回滚干净，避免状态不一致。

#### 命中路径

新请求到达后，服务层执行
`adapter.build_query -> cache_service.lookup -> adapter.apply_resume`：

1. 等待 `vector_update_idle`——确保上一笔异步 save 的向量 upsert 已落库。
2. 调用 Cacheseek lookup：对新 prompt 编码，在向量检索库中查 top-k 近似
   缓存；可选用 Qwen3-VL-Reranker 重排，跟阈值比对决定是否命中。
3. 命中后从 KV 读出 `skip_step` 对应的 latent 张量，封装成 `CacheResult` 返回。

Pipeline 拿到的 `latent_data` 字典里包括 `cached_latent`、`skip_step`、
`saved_steps`。Pipeline 于 `skip_step` 处重启去噪循环，并按 `saved_steps`
把当次的 latent 也快照下来——缓存就是这样越攒越多的。

### 缓存参数

`VideoBasedApproximateCache` 关心的核心参数：

| 参数                           | 类型    | 描述                                 |
| ---------------------------- | ----- | ---------------------------------- |
| `key_steps`                  | list  | pipeline 被要求 snapshot 的 step 列表    |
| `video_similarity_threshold` | float | 向量搜索的命中下限                          |
| `rerank_enabled`             | bool  | 是否启用 Qwen3-VL-Reranker 在 top-k 上重排 |
| `rerank_top_k`               | int   | 进入 rerank 的候选数量                    |
| `rerank_score_threshold`     | float | rerank 启用时的命中下限                    |
| `video_embedding_max_frames` | int   | 视频编码时最多采样的帧数                       |
| `video_vector_collection`    | str   | FAISS collection 名（默认 `video`）     |

> 命中后到底从第几步重启由 `_determine_skip_step` 决定：当前实现里 `similarity`高于 rerank 阈值且 `5 ∈ saved_steps` 时跳过到第 5 步，否则视为 miss。需要自定义跳点策略时可在子类里覆盖此方法。

### 使用 VideoBasedApproximateCache

在 pipeline 文件里声明 `CACHE_CONFIG` 即可启用（service 启动时由
`CacheServiceFactory` 自动加载）：

```python
# examples/wan_video/wan22_14b_text_to_video_service.py
CACHE_CONFIG = dict(
    enable_latent_cache=True,
    latent_cache_dir=os.getenv("TELEFUSER_LATENT_CACHE_DIR", "./latent_cache/wan22_t2v"),
    cache_mode="write_only",
    kv_store_type="local_file",
    vector_store_type="faiss",
    # Qwen3-VL-Embedding-2B hidden_size=2048，必须与 vector_store 维度一致。
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


---

## 使用示例脚本

| Pipeline              | 脚本                                                              | 说明                   |
| --------------------- | --------------------------------------------------------------- | -------------------- |
| Wan2.2 14B T2V（启用缓存）  | `examples/wan_video/wan22_14b_text_to_video_service.py`         | 完整 latent cache 配置示例 |
| Wan2.2 14B T2V（不启用缓存） | `examples/wan_video/wan22_14b_text_to_video_service_nocache.py` | 同样接口、关闭缓存对照组         |

启动服务：

```bash
telefuser serve examples/wan_video/wan22_14b_text_to_video_service.py \
    --port 8000 \
    --enable-latent-cache \
    --cache-mode read_write
```

---

## CacheConfig 字段说明

完整定义在 `telefuser/cache_mem/config.py`，下表给出默认值。

### 基础

| 字段 | 默认 | 说明 |
|---|---|---|
| `enable_latent_cache` | `False` | 总开关，CLI `--enable-latent-cache` 会翻它。 |
| `cache_mode` | `READ_WRITE` | `READ_WRITE` / `READ_ONLY` / `WRITE_ONLY`。 |
| `latent_cache_dir` | `./latent_cache` | 存储、metadata、FAISS、日志的根目录。 |
| `max_cache_size_gb` | `10` | 软淘汰上限（按 `last_access_time` 做 LRU）。 |

### 日志

| 字段 | 默认 |
|---|---|
| `cache_log_enabled` | `True` |
| `cache_log_dir` | `{latent_cache_dir}/logs` |
| `cache_log_level` | `DEBUG` |
| `cache_log_rotation` | `100 MB` |
| `cache_log_retention` | `7 days` |

### KV / Vector 后端

| 字段 | 默认 | 说明 |
|---|---|---|
| `kv_store_type` | `local_file` | 或 `fluxon`（stub）。 |
| `vector_store_type` | `faiss` | 或 `qdrant`（stub）。 |
| `vector_dim` | `2048` | 必须和 embedder 输出维度一致。 |
| `faiss_index_dir` | `{latent_cache_dir}/faiss` | |
| `qdrant_url` / `qdrant_api_key` | `""` / `None` | 等真正接 Qdrant 时再配。 |
| `cache_strategy_type` | `video_approximate` | 策略注册表里的 key。 |

### 策略与 embedding

| 字段 | 默认 | 说明 |
|---|---|---|
| `key_steps` | `[0, 1, 2, 3, 4, 5]` | pipeline 被要求 snapshot 的 step 列表。 |
| `lookup_mode` | `video` | |
| `video_embedding_enabled` | `True` | |
| `video_embedding_model_path` | `Qwen/Qwen3-VL-Embedding-2B` | |
| `video_embedding_max_frames` | `16` | |
| `video_embedding_fps` | `1.0` | |
| `text_embedding_model_path` | `""` | 留空则复用 video embedder。 |
| `video_similarity_threshold` | `0.10` | 向量搜索的命中下限。 |
| `rerank_enabled` | `False` | 开了就用 Qwen3-VL-Reranker 在 top-k 上重排。 |
| `rerank_top_k` | `5` | |
| `rerank_score_threshold` | `0.90` | rerank 启用时的命中下限。 |

当 text 和 video 的 embedding 配置最终落到同一个 model + device 时，
`VideoBasedApproximateCache` 会让两者共享同一个 `Qwen3VLEncoder` 实例，
省下大约 5 GB 显存和一次冷加载。

### 异步保存

| 字段 | 默认 | 说明 |
|---|---|---|
| `save_async_enabled` | `True` | 把 `save` 卸到 worker 线程。 |
| `save_queue_size` | `2` | `0` 表示不限。 |
| `save_on_full` | `drop` | `drop` / `sync` / `downgrade`（downgrade 是 TODO）。 |
| `save_queue_warn_threshold` | `8` | 队列深度超此值打 warning。 |
| `vector_wait_warn_s` | `2.0` | `lookup` 等向量栅栏超过此值打 warning。 |
| `vector_wait_timeout_s` | `120.0` | 等到 timeout 就放弃栅栏，按 miss 走。 |
| `flush_on_shutdown` | `True` | `CacheService.shutdown` 会先把队列里的任务放空。 |

### Cache mode 三档

| 模式 | 效果 |
|---|---|
| `READ_WRITE` | lookup 命中、写完也回写。常态。 |
| `READ_ONLY` | lookup 命中、但不更新缓存。在线灰度期间用得上。 |
| `WRITE_ONLY` | lookup 永远 miss、只攒缓存。对着 benchmark 跑一遍预热 cache 时常用。 |

---

## 架构总览

```
┌────────────────────────────────────────────────────────┐
│                  LatentCache（外观类）                  │
│                                                        │
│  ├─ Strategy        VideoBasedApproximateCache          │
│  │     ├─ prompt_encoder  Qwen3-VL-Embedding           │
│  │     ├─ video_encoder   Qwen3-VL-Embedding（共享）    │
│  │     └─ reranker        Qwen3-VL-Reranker（可选）     │
│  │                                                     │
│  ├─ KVStore         LocalFileKVStore | FluxonKVStore*  │
│  ├─ VectorStore     FAISSVectorStore | QdrantStore*    │
│  └─ MetadataManager LocalCacheMetadataManager          │
└──────────▲─────────────────────────────────────────────┘
           │ 通过 CacheService（异步写回包装）
           │
   FastAPI 请求线程 / pipeline
```

`*` 标注的 Fluxon / Qdrant 后端目前是 stub（`NotImplementedError`），
线上路径只有 `LocalFileKVStore` + `FAISSVectorStore` 两个分支。
