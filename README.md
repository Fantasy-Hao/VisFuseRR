# VisFuseRR

给定文本查询和所属文档，分别编码页面图像和页面文本，融合两路分数后召回候选页。逐点重排按单页相关性评分，列表式重排联合比较候选，输出页序、证据框和理由。

## 数据集

| --dataset-name | 默认目录与输入 | 评测标注 |
|---|---|---|
| `mmdocir` | `data/mmdocir/MMDocIR_annotations.jsonl`、`MMDocIR_pages.parquet`，定位评估另读取 `MMDocIR_layouts.parquet` | 证据页、区域框 |
| `docscope` | `data/docscope/benchmark.json` 和 `pdfs/` | 可回答查询的证据页、区域框、事实证据 |
| `longdocurl` | `data/longdocurl/` 下的 `LongDocURL_public_with_subtask_category.jsonl` 或 `LongDocURL_public.jsonl`，以及 `pdfs/` 或 `pdf_files.tar.gz` | 非空证据页，及可可靠映射到页面的区域框 |

三套数据的候选范围均为查询所属文档的全部页面。LongDocURL 优先读取带子任务的标注文件。从压缩包读取时，程序按需将 PDF 解到 `pdfs/`。文本路使用 PDF 的原生文本层，空文本按占位符编码。DocScope 默认使用 `test`。如果用 `dev` 调参，可以用 `--split test-unseen` 排除与开发集共用 PDF 的测试查询。

## 编码、检索和重排

环境要求 Python 3.10 或更高（代码使用 `zip(..., strict=True)`）。在项目根目录运行以下命令。脚本会设置模块搜索路径。直接调用 Python 时，可先设置 `PYTHONPATH="$PWD/code"`，再运行 `python -m retriever.encode` 等模块入口。将 Qwen3-VL-Embedding-8B 和 Qwen3-VL-Reranker-8B 的权重放入 `model/<模型名>/`。可用 `--embedding-model` 或 `--reranker-model` 指定本地目录。

```bash
python -m pip install -r requirements-qwen.txt
bash run.sh encode --dataset-name mmdocir
bash run.sh retrieve --dataset-name mmdocir --fusion sigmoid-zscore --alpha 0.60 --topk 50
bash run.sh rerank pointwise --dataset-name mmdocir --topk 10
```

- `bash run.sh --help` 显示各阶段入口。`bash run.sh pointwise` 和 `bash run.sh rerank pointwise` 运行同一流程；`bash run.sh rerank` 省略方法时默认 pointwise。
- 改用其他数据集时，给各阶段传入相同的 `--dataset-name` 和适用的 `--split`、`--dataset`、`--limit`。快速检查可给各阶段加 `--limit 2 --output output/smoke`，评估文件会写到 `results/smoke/`。
- 用 `--alpha 0`、`--alpha 1` 分别跑文本和图像单路。`--grid-search` 在本次选择的查询上拟合全局图像权重，独立评测时使用预先选定的 `--alpha`。

## 本地 Listwise 重排

在 Linux 服务器上准备完整的 `Qwen/Qwen3.8-27B-FP8` 权重，放在 `model/Qwen3.8-27B-FP8/`。若权重放在项目根目录的 `Qwen3.8-27B-FP8/`，脚本也会识别。用独立的 Python 环境安装 vLLM，避免和编码、逐点重排环境的 PyTorch 版本冲突。下面以支持 CUDA 13.0 的 NVIDIA 驱动为例。[官方 FP8 权重](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · [vLLM GPU 安装文档](https://docs.vllm.ai/en/v0.34.0/getting_started/installation/gpu/) · [Qwen3.8-27B 部署文档](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

```bash
python3.12 -m venv .venv-listwise
source .venv-listwise/bin/activate
python -m pip install -r requirements-server.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
vllm download-kernels
CUDA_VISIBLE_DEVICES=0 bash run.sh serve
```

服务脚本按单张 RTX PRO 6000 96 GB 配置。vLLM 从检查点识别 FP8 量化，计算 dtype 为 BF16。上下文上限为 131,072 token，默认 2 条并发，每条最多 10 张图，显存利用率为 0.90。可用 `QWEN38_MODEL_DIR` 和 `QWEN38_PORT` 指定路径和端口。

服务就绪后，在同一台服务器的另一个终端激活原项目环境，用已有检索产物运行重排。以下命令先跑 3 条，确认正常后可省略 `--limit`。

```bash
bash run.sh listwise --dataset-name mmdocir --topk 10 --limit 3 \
  --candidates output/retriever/mmdocir/Qwen3VLEmb8B/retrieval/topk_candidates.jsonl \
  --output output/listwise-local-smoke
```

客户端只连接本机回环地址，默认使用 `http://127.0.0.1:8000/v1`。自定义端口时传 `--base-url`。默认提交完整 OCR 和原尺寸页面图像。思考预算为 32,768 token，最终答案预留 16,384 token，生成总预算为 49,152 token，可给输入预留 81,920 token。预算可由 `--thinking-budget` 和 `--max-tokens` 调整。读取超时默认 1,800 秒，可用 `--read-timeout` 修改。

每条结果记录 `finish_reason`、`usage`、请求耗时和排序是否修补。达到生成上限的结果回退到原检索序。排序指标包含回退结果，`metrics_rerank.json` 的 `run` 字段单独统计实际完成重排、回退和排序修补的查询数量。模型加载和上下文限制由 vLLM 处理。

Listwise 默认支持断点续跑。每条成功的响应会立即写入重排输出目录下的 `checkpoint.sqlite3`。中断后在服务可用时重跑原命令并保留同一个 `--output`，程序会复用相同输入和推理参数的结果，重试失败或未完成的查询。全部处理完后会重新生成 `rerank_scores.jsonl` 和评估指标。更换后端权重或需要全新实验时，使用新的 `--output`，并通过 `--candidates` 指向已有候选文件。

## 产物与指标

| 阶段 | `output/` 推理产物 | `results/` 评估产物 |
|---|---|---|
| 编码 | `retriever/<数据集>/<编码器>/encode/embeddings.npz` | — |
| 检索 | `retriever/<数据集>/<编码器>/retrieval/topk_candidates.jsonl` | 同级相对目录下的 `metrics_retrieval.json`，开启网格搜索时另有 `gridsearch.json` |
| 重排 | `reranker/<数据集>/<编码器>/rerank/<方式>/<模型>/rerank_scores.jsonl` | 同级相对目录下的 `metrics_rerank.json`，Listwise 另有 `metrics_localization.json` |

`--output output/<实验名>` 对应 `results/<实验名>`。其他输出路径以其目录名在 `results/` 下建立子目录。重复运行同一配置会覆盖对应文件，做对照时可给每组指定独立的 `--output`。

`Hit@K` 表示至少命中一个证据页，`Recall@K` 表示命中的证据页比例，`RecallAll@K` 表示命中全部证据页。`IoU@0.5` 在召回且带有效框的标注页上计算，`end_to_end_IoU@0.5` 将未召回的标注页计为失败。DocScope 另报 `fact_coverage@0.5`，每条事实命中至少一个可接受的区域即记为覆盖。

历史 MMDocIR 实验见[报告索引](report/README.md)。报告记录当时的 API 实现与自适应加权等实验；当前 CLI 使用本地权重和全局图像权重，不包含这些历史功能。
