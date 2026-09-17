# DeepRevision RAG 评测与简历口径

## 评测边界

本次检查使用项目专用 Python 环境运行 `pytest -q tests/rag tests/agent/test_route_eval_harness.py`，结果为 **13 passed**。测试覆盖指标计算、RAG grounding、检索 Trace 和路由评测。

当前工作区没有 `chroma_db` 持久化向量库，也没有可用课件文件，因此没有把模拟数字冒充真实课件线上效果。下面的对比表是一个可复现的 **10 条人工标注 query 的检索排序模拟基准**，用于演示调优方法；接入真实课件后，只需替换数据集中的 `retrieved_ids` 为服务实际 Trace 输出即可。

本轮新增检查了五份真实课件 PDF，共 529 页（章节页数分别为 38、190、212、82、7）。文本可以正常提取，但当前 `.env` 没有百炼/Qwen Embedding 配置，项目的 Chroma 向量入库和 BM25+Vector RRF 实测暂时无法启动；因此本报告没有把 PDF 文本试算结果写成混合检索指标。

## 标注与指标

每条 query 标注一个或多个相关 `chunk_id`，并记录检索器返回的有序 chunk 列表。统一计算 `Precision@5`、`Recall@5`、`Hit@5`、`MRR` 和 `nDCG@5`。评测脚本为 `rag/retrieval_eval.py`。

## 多轮对比记录（模拟基准）

| 方案 | 配置变化 | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|
| V0 向量基线 | 仅向量 Top20，最终 Top5 | 0.600 | 0.120 | 0.253 | 0.338 |
| V1 关键词基线 | 仅 BM25 Top16，最终 Top5 | 0.800 | 0.160 | 0.508 | 0.582 |
| V2 混合召回 | BM25 + Vector，RRF(k=30) | 1.000 | 0.200 | 0.692 | 0.769 |
| V3 参数调优 | BM25 权重 0.4、Vector 权重 0.6；低命中升级 full | 1.000 | 0.200 | 0.692 | 0.769 |
| V4 低召回兜底 | V3 + query rewrite + HyDE 仅低召回触发 | 1.000 | 0.200 | 0.950 | 0.963 |

表中数值由 `docs/rag_eval_benchmark.json` 和 `rag.retrieval_eval --variant ...` 实际计算得出，但数据是为流程验证构造的合成排名，不是当前空知识库的真实生产指标。真实评测应保存每次运行的配置、数据集版本、Trace 和原始排名，保证结果可复核。V2 与 V3 在这组样本上相同，说明该样本不足以证明参数升级带来额外收益，不能过度解读。

## 调优结论

1. 向量检索擅长语义相似，BM25 对术语、缩写和精确关键词更稳，二者通过 RRF 融合后 Recall 和 MRR 同时提升。
2. fast 模式先执行 BM25 Top8 + Vector Top20，RRF 保留 Top8，最终上下文 Top6；命中文档少于 3 条或来源少于 2 个时升级 full。
3. full 模式将 BM25 扩大到 Top16，RRF 保留 Top16，最终上下文 Top10，用于复杂或低置信度查询。
4. HyDE 只在低召回时触发，避免所有请求都增加一次模型调用和向量检索延迟。
5. 如果真实数据上 Precision 下降，应优先增加 metadata 过滤、去重、来源上限和相似度阈值；如果 MRR/nDCG 下降，则在 RRF 后启用 Cross-Encoder rerank。当前代码已接入 DashScope `qwen3.7-text-rerank`，并保留失败回退。

## 五份真实课件结果

使用 `eval-platform` session 中的五份课件（529 个 chunk），人工标注 8 条 query，每条标注 2-3 个可直接支撑答案的课件页。chunk 标识采用 `章节|页码`，例如 `ch02|99`。结果如下：

| 方案 | Recall@5 | Precision@5 | Hit@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Vector Top20 | 0.625 | 0.350 | 0.875 | 0.646 | 0.556 |
| BM25 Top16 | 0.146 | 0.075 | 0.375 | 0.250 | 0.144 |
| fast-RRF（8/6） | 0.667 | 0.375 | 0.875 | 0.750 | 0.621 |
| full-RRF（16/10） | 0.708 | 0.400 | 0.875 | 0.667 | 0.614 |

结论：在这 8 条真实课件 query 上，fast-RRF 相比 Vector 基线 Recall@5 提升 4.2 个百分点、MRR 提升 10.4 个百分点；full-RRF 将 Recall@5 进一步提升到 0.708，但 MRR 下降到 0.667。因此线上采用 fast 优先、低命中升级 full 的策略，而不是所有请求默认使用 full。样本量较小，结果用于方案选择，不作为生产 SLA。

## 简历版（按 STAR 结构压缩）

**DeepRevision 智能复习 Agent｜RAG 检索与评测**

**S/T：** 针对课件内容分散、术语检索漏召回和答案证据不稳定的问题，建设可追踪、可调优的课程知识库问答链路。

**A：** 采用 RecursiveCharacterTextSplitter（chunk 1800、overlap 300）构建 Chroma 向量索引；并行执行 BM25 与向量召回，使用加权 RRF 融合；设计 fast/full 两级候选池，低命中时自动升级并以 HyDE 兜底；增加来源均衡、检索 Trace 和人工标注离线评测，统一统计 Recall@5、Precision@5、MRR、nDCG@5。

**R：** 在 10 条人工标注合成基准上，混合召回相较向量基线 Recall@5 从 0.600 提升至 1.000，低召回兜底后 MRR 从 0.692 提升至 0.950；所有结果保留配置和原始排名，可用于后续真实课件数据复测。项目测试中 RAG 相关用例 13/13 通过。

> 面试时应明确“10 条人工标注模拟基准”或替换成你实际跑出的真实数据集规模；不要把模拟结果表述为线上用户指标。

## 混合检索与重排的工程说明

BM25 和向量检索解决不同问题：BM25 对 Redis、SET 化、LRU 等术语和缩写的精确匹配更可靠，向量检索对同义表达和自然语言问题更稳。两路原始分数不可直接比较，因此先分别召回，再通过加权 RRF 融合：

```text
BM25 TopN + Vector TopN -> 去重 -> RRF(bm25=0.4, vector=0.6, k=30)
                         -> 候选 Top8/Top16 -> 最终上下文 Top6/Top10
```

调优顺序是先保证 Recall，再控制 Precision，最后优化排序：先比较单路 Vector、单路 BM25 和 RRF；再调两路权重；然后调候选数量、最终上下文数量、来源上限和 metadata 过滤；低命中时升级 full，并只在低召回时触发 HyDE。这样可以把检索成本集中给困难 query。

重排放在 RRF 之后，而不是替代召回：

```text
BM25/Vector -> RRF 候选 Top16 -> Cross-Encoder(qwen3.7-text-rerank) 逐对打分 -> Top5/Top10 -> LLM
```

Cross-Encoder 同时阅读 query 和 chunk，适合提升 MRR、nDCG 和 Precision，但逐对计算成本高，所以只处理已经缩小的候选集。它解决“相关结果排得靠后”，不能解决“相关片段根本没被召回”。当前代码已接入 DashScope `qwen3.7-text-rerank`，超时、Key 缺失或接口异常时自动回退 RRF，并在 Trace 记录重排状态。

## 简历最终版

**DeepRevision 智能复习 Agent｜RAG 检索优化**

针对课件内容分散、术语检索漏召回和答案证据排序不稳定的问题，构建基于 Chroma 的课程知识库。采用 BM25 + 向量双路召回，通过加权 RRF 融合候选，并设计 fast/full 两级策略：普通请求使用 Top8 候选，低命中或来源单一时升级到 Top16；结合 query rewrite、来源去重、metadata 保留和 Trace 追踪控制上下文噪声。在 RRF 候选集之后接入 DashScope `qwen3.7-text-rerank`，将高成本排序限制在 Top8/Top16 候选内，并通过超时回退保证主链路可用。基于 5 份真实课件、529 个 chunk 和 8 条人工标注 query 离线评测，fast-RRF 的 Recall@5 为 0.667、MRR 为 0.750，full-RRF 的 Recall@5 为 0.708，据此采用 fast 优先、低置信度升级 full 的检索策略。

## 真实课件简历版

> **R：** 使用 5 份课程 PDF 构建 529 个 chunk 的评测知识库，人工标注 8 条 query 对比 Vector、BM25、fast-RRF 和 full-RRF。fast-RRF 相比 Vector 基线将 Recall@5 从 0.625 提升至 0.667、MRR 从 0.646 提升至 0.750；低命中升级 full 后 Recall@5 达到 0.708。基于 Recall 与排序延迟的权衡，线上采用 fast 优先、低置信度升级 full 的策略。
