# VisFuseRR

给定文本查询和所属文档，分别编码页面图像和页面文本，融合两路分数后召回候选页。逐点重排按单页相关性评分，列表式重排联合比较候选，输出页序、证据框和理由。

## 数据集

| --dataset-name | 默认目录与输入 | 评测标注 |
|---|---|---|
| `mmdocir` | `data/mmdocir/MMDocIR_annotations.jsonl` 和 `MMDocIR_pages.parquet` | 证据页、区域框 |
| `docscope` | `data/docscope/benchmark.json` 和 `pdfs/` | 可回答查询的证据页、区域框、事实证据 |
| `longdocurl` | `data/longdocurl/` 下的 `LongDocURL_public_with_subtask_category.jsonl` 或 `LongDocURL_public.jsonl`，以及 `pdfs/` 或 `pdf_files.tar.gz` | 非空证据页，及可可靠映射到页面的区域框 |

三套数据的候选范围均为查询所属文档的全部页面。LongDocURL 优先读取带子任务的标注文件。从压缩包读取时，程序按需将 PDF 解到 `pdfs/`。文本路使用 PDF 的原生文本层，空文本按占位符编码。DocScope 默认使用 `test`。如果用 `dev` 调参，可以用 `--split test-unseen` 排除与开发集共用 PDF 的测试查询。

## 编码、检索和重排

在项目根目录设置模块搜索路径。

```bash
export PYTHONPATH="$PWD/code"
```

```powershell
$env:PYTHONPATH = "$PWD/code"
```

下面以 LongDocURL 和 Qwen 模型为例。将模型权重放入 `model/<模型名>/`，也可用 `--embedding-model` 或 `--reranker-model` 指定本地目录。

```bash
python -m pip install -r requirements-qwen.txt
python -m retriever.encode --dataset-name longdocurl
python -m retriever.search --dataset-name longdocurl --fusion sigmoid-zscore --alpha 0.60 --topk 50
python -m reranker.pointwise --dataset-name longdocurl --topk 10
python -m reranker.listwise --dataset-name longdocurl --topk 10
```

最后一条命令需要在环境变量或根目录的 `.env` 中设置 `SILICONFLOW_API_KEY`。可用 `SILICONFLOW_MODEL`、`SILICONFLOW_BASE_URL` 或 `--model` 配置服务与模型。

- 改用其他数据集时，给各阶段传入相同的 `--dataset-name` 和适用的 `--split`、`--dataset`、`--limit`。快速检查可用 `--limit 2 --output output/smoke`。
- 编码器可通过 `--embedding-model-name` 选择 `Qwen3-VL-Embedding-8B`、`jina-embeddings-v4` 或 `WeMM-Embedding-9B`，对应 `requirements-qwen.txt`、`requirements-jina.txt` 和 `requirements-wemm.txt`。检索和重排使用同一编码器名以找到对应产物。
- 逐点模型通过 `--reranker-model-name` 选择 `Qwen3-VL-Reranker-8B` 或 `jina-reranker-m0`。不同模型的依赖版本有差异，可分别安装在对应的虚拟环境。
- 用 `--alpha 0`、`--alpha 1` 分别跑文本和图像单路。`--grid-search` 在本次选择的查询上拟合全局图像权重，独立评测时使用预先选定的 `--alpha`。

## 产物与指标

- `output/retriever/` 保存嵌入、`topk_candidates.jsonl` 和 `metrics_retrieval.json`。重复运行同一数据集和编码器会覆盖对应的检索产物，消融对照可为每组指定独立的 `--output`。
- `output/reranker/` 保存重排结果和前后对照指标。`Hit@K` 表示至少命中一个证据页，`Recall@K` 表示命中的证据页比例，`RecallAll@K` 表示命中全部证据页。固定 Top-10 候选后，重排只改变页的次序。
- 列表式重排另保存 `metrics_localization.json`。`IoU@0.5` 在召回且带有效框的标注页上计算，`end_to_end_IoU@0.5` 将未召回的标注页计为失败。DocScope 另报 `fact_coverage@0.5`，每条事实命中至少一个可接受的区域即记为覆盖。

历史 MMDocIR 实验见[报告索引](report/README.md)。
