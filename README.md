# VisFuseRR 多模态文档检索与重排

项目分别编码查询、页面文本和页面图像，融合图文相似度生成候选页，再对同一候选列表做逐点或列表式重排。列表式重排还输出归一化的证据框、证据类型、相关性档位和理由。

## 项目结构

```text
code/
  common/
    dataset.py    数据集注册、命令行选项和数据检查
    mmdocir.py    MMDocIR 页面、查询和布局标注
    docscope.py   DocScope 划分、PDF 页面和事实证据映射
    vidore_v3.py  ViDoRe V3 子库、查询语言、分级 qrels 和证据框
    utils.py     共用坐标、图像、路径和 I/O 工具
  retriever/      模型、编码、检索融合和排序指标
  reranker/       逐点重排、列表式重排和定位指标
data/             本地数据集
model/            本地模型权重与配置
output/           编码、检索、重排和评测产物
report/           历史实验报告
```

## 数据集与评测范围

| 数据集 | 默认目录与文件 | 默认范围 |
|---|---|---|
| MMDocIR | `data/mmdocir/` 下的 `MMDocIR_annotations.jsonl`、`MMDocIR_pages.parquet`、`MMDocIR_layouts.parquet` | 1,658 条查询。每条查询在其所属文档的整份页面范围内检索。页面文本来自数据集 OCR。 |
| DocScope | `data/docscope/` 下的 `benchmark.json` 和 `pdfs/*.pdf` | `--split test`，本地公开版有 678 条可回答测试查询。每条查询可搜索所属 PDF 的全页。页图按 144 DPI 渲染，文本直接从 PDF 文本层提取，缺少文本层的页面以空文本处理。 |
| ViDoRe V3 | `data/vidore_v3/vidore_v3_<子集>/` 下的 `corpus/`、`queries/`、`qrels/` 中的 `test-*.parquet` | `--subset all --language native`，本地 8 个公开子集共 2,419 条原语言查询。每条查询搜索其所在子集的全部页面，文本来自数据集的 OCR Markdown。 |

- DocScope 支持 `dev`、`test`、`test-unseen`、`all`。`test-unseen` 去掉测试集中出现在开发集的 PDF，在当前发布版中有 656 条可回答查询。`all` 合并开发和测试的可回答查询。常规检索分母仅包含可回答查询。
- ViDoRe V3 子集可选 `all`、`computer_science`、`energy`、`finance_en`、`finance_fr`、`hr`、`industrial`、`pharmaceuticals`、`physics`。语言可选 `native`、`all`、`english`、`french`、`german`、`italian`、`portuguese`、`spanish`。`native` 在 Energy、Finance FR、Physics 使用法语，其余子集使用英语。`all` 语料选项覆盖本地可用子集，`all` 语言选项包含全部翻译。用 `--dataset` 指定数据根目录，或者配合具体 `--subset` 指向单个子集目录。
- `--limit N` 按当前数据选择取前 N 条查询，候选页范围仍是完整文档或完整子集。对 ViDoRe V3 做小样本检查时建议同时传 `--subset`，以便明确测试的语料范围。

## 环境准备

以下命令在项目根目录用 PowerShell 执行。三个环境分别使用对应的 `requirements`，以便不同版本的 Transformers 共存。`python` 指向可用的 Python 3.10 或更新版本。

```powershell
$env:PYTHONPATH = Join-Path $PWD 'code'

python -m venv .venv/qwen
python -m venv .venv/jina
python -m venv .venv/wemm

$qwen = '.\.venv\qwen\Scripts\python.exe'
$jina = '.\.venv\jina\Scripts\python.exe'
$wemm = '.\.venv\wemm\Scripts\python.exe'

& $qwen -m pip install -r requirements-qwen.txt
& $jina -m pip install -r requirements-jina.txt
& $wemm -m pip install -r requirements-wemm.txt
```

本地模型从 `model/<模型名>/` 读取权重。编码可用 `--embedding-model` 指定其他路径，逐点重排可用 `--reranker-model`。`--device auto` 会在 PyTorch 可用 CUDA 时使用 GPU，CPU 环境可显式指定 `--device cpu`。运行前请确认 PyTorch 可识别 GPU 且显存能容纳模型权重与输入。`--batch-size 1` 能减少批处理显存，模型权重仍需要完整加载。

列表式模型通过 SiliconFlow API 调用。在根目录 `.env` 中配置 `SILICONFLOW_API_KEY`。`SILICONFLOW_BASE_URL` 和 `SILICONFLOW_MODEL` 可选，默认分别为 `https://api.siliconflow.cn/v1` 和 `Qwen/Qwen3.8-27B`。命令中的 `--model` 优先于 `.env` 模型设置。列表式调用会将所选候选页的图像及文本发送给 API。[SiliconFlow 视觉输入](https://docs.siliconflow.cn/docs/userguide/capabilities/vision)

## 检查本地数据

以下检查读取查询、真值和一张样例图，不加载模型。纯数据检查也可以使用装有 `pyarrow`、`Pillow`、`PyMuPDF` 的 Python 环境。

```powershell
& $qwen -m common.dataset --dataset-name mmdocir --check-layouts
& $qwen -m common.dataset --dataset-name docscope --split test --check-layouts
& $qwen -m common.dataset --dataset-name docscope --split test-unseen --check-layouts
& $qwen -m common.dataset --dataset-name vidore_v3 --subset all --language native
& $qwen -m common.dataset --dataset-name vidore_v3 --subset finance_en --language native --limit 2 --check-layouts
```

## 三个数据集的检索命令

先选一组 `$dataset` 参数。下面每条模型命令都可用于这三个数据集。ViDoRe V3 若只跑某个正式子集，将 `all` 换成相应子集名。DocScope 若用开发集调参，将正式测试的 `test` 换成 `test-unseen` 可按 PDF 隔离。

```powershell
# MMDocIR
$dataset = @('--dataset-name', 'mmdocir')

# DocScope
$dataset = @('--dataset-name', 'docscope', '--split', 'test')

# ViDoRe V3 全部本地公开子集，使用各自原语言
$dataset = @('--dataset-name', 'vidore_v3', '--subset', 'all', '--language', 'native')
```

同一个数据集和嵌入模型先编码，再检索。检索命令使用查询内 Sigmoid–Z-score 校准和默认图像权重 0.60。省略 `--fusion sigmoid-zscore` 时使用原始分数线性融合。

```powershell
# Qwen3-VL-Embedding-8B
& $qwen -m retriever.encode @dataset --embedding-model-name Qwen3-VL-Embedding-8B --batch-size 1
& $qwen -m retriever.search @dataset --embedding-model-name Qwen3-VL-Embedding-8B --fusion sigmoid-zscore --topk 50

# jina-embeddings-v4
& $jina -m retriever.encode @dataset --embedding-model-name jina-embeddings-v4 --batch-size 1
& $jina -m retriever.search @dataset --embedding-model-name jina-embeddings-v4 --fusion sigmoid-zscore --topk 50

# WeMM-Embedding-9B
& $wemm -m retriever.encode @dataset --embedding-model-name WeMM-Embedding-9B --batch-size 1
& $wemm -m retriever.search @dataset --embedding-model-name WeMM-Embedding-9B --fusion sigmoid-zscore --topk 50
```

检索只读取已保存的向量，可以在包含 NumPy 和数据集依赖的任一环境中执行。`--grid-search` 在当前查询上按平均 Recall@5 搜索一个全局融合权重。这一模式适用于开发或诊断，同一批查询上的调参结果应作为样本内结果报告。

## 三个数据集的重排命令

沿用编码与检索时的 `$dataset` 参数。`$embedding` 指定要重排哪一个检索器的候选，可替换为上节任一嵌入模型。两个逐点重排器各用自己的环境，列表式命令可在任一已安装 HTTP 依赖的环境中运行。

```powershell
$embedding = 'Qwen3-VL-Embedding-8B'  # 或 jina-embeddings-v4、WeMM-Embedding-9B

# Qwen3-VL-Reranker-8B，逐点查询 + 页图
& $qwen -m reranker.pointwise @dataset --embedding-model-name $embedding --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 1

# jina-reranker-m0，逐点查询 + 页图
& $jina -m reranker.pointwise @dataset --embedding-model-name $embedding --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 1

# Qwen/Qwen3.8-27B，列表式查询 + 页图 + 页面文本
& $qwen -m reranker.listwise @dataset --embedding-model-name $embedding --model 'Qwen/Qwen3.8-27B' --topk 10
```

- 检索和重排的 `--topk` 可不同，检索结果需覆盖重排的候选数。`--limit` 如用于小样本检查，建议在三个阶段传同一值，并用同一个独立 `--output output/smoke` 避免覆盖正式结果。
- 列表式支持 `--no-text`、`--no-image`、`--no-thinking`、`--max-image-side`、`--max-ocr-chars`、`--thinking-budget` 和 `--max-tokens`。支持相同模型下的图像单模态或文本单模态评估。文本单模态只计算排序指标。
- Linux 或 Git Bash 可以激活相应环境，使用现有的 `bash code/retriever/run_encode.sh`、`bash code/retriever/run_retrieve.sh`、`bash code/reranker/run_pointwise.sh`、`bash code/reranker/run_listwise.sh`，参数与 Python 入口一致。

## 产物与指标

默认路径中的 `<数据选择>` 为 `mmdocir`、`docscope/<split>` 或 `vidore_v3/<subset>/<language>`。Qwen3-VL-Embedding-8B 的输出目录名为 `Qwen3VLEmb8B`。含 `/` 的重排模型名会进行 URL 编码。

```text
output/retriever/<数据选择>/<嵌入模型>/encode/embeddings.npz
output/retriever/<数据选择>/<嵌入模型>/retrieval/topk_candidates.jsonl
output/retriever/<数据选择>/<嵌入模型>/retrieval/metrics_retrieval.json
output/reranker/<数据选择>/<嵌入模型>/rerank/<方式>/<重排模型>/rerank_scores.jsonl
output/reranker/<数据选择>/<嵌入模型>/rerank/<方式>/<重排模型>/metrics_rerank.json
output/reranker/<数据选择>/<嵌入模型>/rerank/listwise/<重排模型>/metrics_localization.json
```

- `topk_candidates.jsonl` 是检索与重排的交接文件，包含页 UID 和分数。候选可用 `--candidates` 指定。检索可用 `--embeddings` 复用已有向量，再用 `--output` 隔离不同融合实验。编码缓存根据数据文件、模型文件及编码设置校验，`--force` 强制重建。与参考答案、证据页和框相关的真值字段不进入重排提示。
- 排序结果有逐查询、总体和分组指标。`Recall@K` 衡量正例页覆盖比例，`Hit@K` 衡量是否命中至少一页，`RecallAll@K` 衡量是否召回全部标注页。ViDoRe V3 的 nDCG 使用 1／2 级相关性作为增益，其他排序指标将两档都计为正例。MMDocIR 和 DocScope 使用二值页相关性。
- 定位结果包含已召回真值页上的 IoU，以及把未召回的有框真值页计为失败的 `end_to_end_IoU@0.5`。DocScope 的 `fact_coverage@0.5` 在任一有效支持框达标时计该事实被覆盖。它度量的是区域覆盖，分母为可映射到有效证据框的事实。DocScope 的 `RecallAll@K` 仍按所有标注页计，含备选证据页时可结合事实覆盖率解读。
- 列表式请求失败时沿用检索序，输出保留 `api_error`。正式汇报前应检查失败数、排序来源和输出截断情况。[历史实验报告索引](report/README.md)
