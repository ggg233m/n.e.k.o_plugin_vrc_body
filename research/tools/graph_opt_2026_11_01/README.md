# 图优化取证脚本（2026-11-01）

配套文档：[`Docs/CPU派发开销剖析-2026-11-01.md`](../../../Docs/CPU派发开销剖析-2026-11-01.md) §13。

**为什么独立成一个目录**：这些脚本原本在 `.tmp/` 里，但 `.tmp/` 被 gitignore，
文档里的复现链会断。归档到这里，保证半年后还能跑。

## 运行环境（照抄，缺一个就会挂）

```powershell
$root = "H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body"
$env:PYTHONUTF8 = "1"                                          # 不设 → GBK 让 triton intel driver 抛 UnicodeDecodeError
$env:TMP = "$root\.tc_cache\tmp"; $env:TEMP = $env:TMP        # 不设 → C 盘只剩 2.3 GB → BrokenProcessPool
$env:TORCHINDUCTOR_CACHE_DIR = "$root\.tc_cache\inductor"
$env:TRITON_CACHE_DIR = "$root\.tc_cache\triton"
$env:TORCHINDUCTOR_COMPILE_THREADS = "1"                       # 多线程会 pass_fds 报错
$env:CC = "icpx"; $env:CXX = "icpx"                           # 不设 → Failed to find C compiler
$env:PATH = "D:\Program Files (x86)\Intel\oneAPI\2026.0\bin;" + $env:PATH

& "$root\.venv_xpu\Scripts\python.exe" <脚本>
```

## 脚本清单

| 文件 | 作用 | 关键输出 |
|---|---|---|
| `_graph_opts.py` | 四种 compile 配置对比 | `RESULT <tag> compile=?s cpu=? wall=? kernel=?` |
| `_final_g1.py` | 完整 `act()` 端到端（预处理+前向+head） | `完整act p50= p95= p99=`、`2Hz 门槛 PASS/FAIL` |
| `_aggr_act.py` | 开 `aggressive_fusion` 的完整 `act()` | `AGGRFUZE_ACT p50= p95= p99=` |
| `_ze_ext4.py` | 枚举 L0 驱动扩展 | `has record_replay_graph = True/False` |
| `_cppwrap_min.py` | `cpp_wrapper` 最小复现 | `CPP_WRAPPER OK/FAIL` + 完整编译器命令行 |
| `_run_cpp3.bat` | 分阶段切编译器（icpx → clang-cl） | — |
| `_ur_info2.py` | 直查 UR 层 `0x2080` | **已知会 access violation，属未完成实验** |
| `_graph_exp.py` / `_graph_expA.py` | 整模型单图捕获（失败复现） | `[A] FAIL ... UNSUPPORTED_FEATURE` |
| `_graph_diag.py` | ViT / LM 分别捕获 | `vit_graph` / `lm_graph` 各 `{"ok": true}` |
| `_dual_graph.py` | 双图串联可行性 + 正确性 | `[6] cos=1.00000000`、`dual_graph_ms` |
| `_adapter_cmp.py` | V2 适配器下 replay 耗时（含 `min` 值） | `vit_ms` / `lm_ms` 的 p50 与 min |
| `_e2e_graph.py` | `PolicyRuntime(graph=True)` 端到端 vs eager | `{graph, eager, cos, max_abs, speedup}` |
| `_latency_vs_compute.py` | 判别 LM 图是「算不动」还是「在等」 | `single/batched/two-stream` 三项证据 |
| `_bottleneck.py` | 分引擎测 compute vs 3d 占用 | `compute_pct` / `render3d_pct` |
| `_isolate.py` | 隔离 ViT/glue/LM，LM 随 token 数变化 | `vit_only` / `glue` / `lm_Ntok` |

## 用法示例

```powershell
# 测 aggressive_fusion
& "$root\.venv_xpu\Scripts\python.exe" research/tools/graph_opt_2026_11_01/_graph_opts.py C_aggrfuse

# 枚举 L0 扩展（验证 A770 报不报 record_replay_graph）
& "$root\.venv_xpu\Scripts\python.exe" research/tools/graph_opt_2026_11_01/_ze_ext4.py
```

## 五个坑（都踩过）

1. **`_ze_ext4.py` 的 struct 布局**：`_ze_driver_extension_properties_t` 的 `version` 字段是
   `uint32`，不是 `char[256]`。写成 `char[256]` 会得到**假阴性**——只解出 6 个乱码名字，
   然后误判成「驱动不支持 graph」。我第一次就栽在这里。
2. **`_graph_opts.py` 的 B_cppwrapper 一定失败**：三个编译器都编不过（icpx 不认 MSVC 参数，
   MSVC 不认 `__attribute__((noinline))`，clang-cl 编不出 Triton 的 `spirv_utils.pyd`）。
   失败是预期的，别以为是环境没配好。
3. **首次编译 6–8 分钟**（cold cache）。后续热缓存 45 秒左右。别中途杀进程，
   会让 inductor 缓存锁留下死等。
4. **`flash-linear-attention` 在 A770 上是负优化，别装。** 图重放场景实测
   LM 62.2 → 367.7 ms（**5.4× 退化**），比 eager 场景的 8% 退化严重得多 ——
   图把那些小 kernel 全固化了。详见文档 §15.13。
5. **跨引擎求和 GPU 占用是错的。** 3d / compute / copy 是独立队列，
   VRChat 占满 3d 不影响 compute。要看 `engtype_compute`，别把 194 个实例加起来。
6. **后端在线（LAS2 worker 活着）时，别在同一块 A770 上跑任何第二个 XPU 推理** ——
   包括本目录的对照脚本。实测两次 DEVICE_LOST 把 LAS2 打崩（exit 0xC0000005），
   本会话内不会重试，只能重启后端。要跑对照脚本先 navmesh/stop。


## 环境快照（出结果时的配置）

- GPU：Intel Arc A770（dg2，512 EU，16 GB）
- 驱动：`32.0.101.8974`（2026-08-11），L0 runtime `1.15.39183+1`
- torch `2.14.1+xpu`，triton `3.8.0`，transformers `5.21.0`
- oneAPI 2026.0，LevelZeroSDK 1.32.0
- Python 3.14.9

换环境后数字会变，**结论也可能变**——尤其是 UR 版本一旦实现 `0x2080`，
§13.1 的判定就要推翻重来。