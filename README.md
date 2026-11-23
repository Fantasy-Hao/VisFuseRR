# VisFuseRR 多模态文档检索与重排

项目分别编码查询、页面文本和页面图像，融合图文相似度生成候选页，再对同一候选列表做逐点或列表式重排。列表式重排还输出归一化的证据框、证据类型、相关性档位和理由。

当前运行入口覆盖预训练模型推理、固定融合权重和全局权重搜索。历史报告中的查询自适应融合属于历史实验，相关命令见[历史实验报告](report/README.md)。

## 项目结构

```text
code/
  dataset/
    __init__.py   数据集选择与共用命令行参数
    mmdocir.py    MMDocIR 页面、查询和布局标注
    docscope.py   DocScope 划分、PDF 页面和事实证据映射
    vidore_v3.py  五个英语子库、分级 qrels 和证据框
  utils.py        图像、坐标、路径和 I/O 工具
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
| MMDocIR | `data/mmdocir/` 下的 `MMDocIR_annotations.jsonl`、`MMDocIR_pages.parquet` | 1,658 条查询。每条查询在其所属文档的整份页面范围内检索。页面文本来自数据集 OCR，证据框来自查询标注。 |
| DocScope | `data/docscope/` 下的 `benchmark.json` 和 `pdfs/*.pdf` | `--split test`，本地公开版有 678 条可回答测试查询。每条查询可搜索所属 PDF 的全页。页图按 144 DPI 渲染，文本直接从 PDF 文本层提取，缺少文本层的页面以空文本处理。 |
| ViDoRe V3 | `data/vidore_v3/vidore_v3_<子集>/` 下的 `corpus/`、`queries/`、`qrels/` 中的 `test-*.parquet` | 默认评测五个英文子库，共 1,489 条英文查询、12,969 页。每条查询搜索其所属子库的全部页面，页面文本来自 OCR Markdown。 |

- DocScope 支持 `dev`、`test`、`test-unseen`、`all`。`test-unseen` 去掉测试集中出现在开发集的 PDF，在当前发布版中有 656 条可回答查询。`all` 合并开发和测试的可回答查询。常规检索分母仅包含可回答查询。
- ViDoRe V3 的 `--subset` 可选 `all`、`computer_science`、`finance_en`、`hr`、`industrial`、`pharmaceuticals`。加载器固定选取英文查询。用 `--dataset` 可指定数据根目录。
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

## 三个数据集的检索命令

先选一组 `$dataset` 参数。下面每条模型命令都可用于这三个数据集。ViDoRe V3 若只跑某个正式子集，将 `all` 换成相应子集名。DocScope 若用开发集调参，将正式测试的 `test` 换成 `test-unseen` 可按 PDF 隔离。

```powershell
$datasets = @{
    mmdocir   = @('--dataset-name', 'mmdocir')
    docscope  = @('--dataset-name', 'docscope', '--split', 'test')
    vidore_v3 = @('--dataset-name', 'vidore_v3', '--subset', 'all')
}
$dataset = $datasets.vidore_v3   # 也可选 $datasets.mmdocir 或 $datasets.docscope
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

检索只读取已保存的向量，可以在包含 NumPy 和数据集依赖的任一环境中执行。`--grid-search` 在当前查询上按平均 Recall@5 搜索一个全局融合权重。开发集调优后，可用 `--alpha` 在测试集应用找到的权重，例如 `--alpha 0.65`。同一批查询上的调参结果应作为样本内结果报告。

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
- 列表式通过 `--max-image-side` 和 `--max-ocr-chars` 控制单页图像和文本输入大小。
- Linux 或 Git Bash 可以激活相应环境，使用现有的 `bash code/retriever/run_encode.sh`、`bash code/retriever/run_retrieve.sh`、`bash code/reranker/run_pointwise.sh`、`bash code/reranker/run_listwise.sh`，参数与 Python 入口一致。

## 产物与指标

默认路径中的 `<数据选择>` 为 `mmdocir`、`docscope/<split>` 或 `vidore_v3/<subset>`。Qwen3-VL-Embedding-8B 的输出目录名为 `Qwen3VLEmb8B`。含 `/` 的重排模型名会进行 URL 编码。

```text
output/retriever/<数据选择>/<嵌入模型>/encode/embeddings.npz
output/retriever/<数据选择>/<嵌入模型>/retrieval/topk_candidates.jsonl
output/retriever/<数据选择>/<嵌入模型>/retrieval/metrics_retrieval.json
output/reranker/<数据选择>/<嵌入模型>/rerank/<方式>/<重排模型>/rerank_scores.jsonl
output/reranker/<数据选择>/<嵌入模型>/rerank/<方式>/<重排模型>/metrics_rerank.json
output/reranker/<数据选择>/<嵌入模型>/rerank/listwise/<重排模型>/metrics_localization.json
```

- 编码会生成新的向量文件。由旧版本迁移时，请按选定的数据集和模型重新编码并检索。`--embeddings` 可指定已编码的向量，`--output` 可隔离不同融合实验。`topk_candidates.jsonl` 是检索与重排的交接文件，包含页 UID 和分数。
- 排序结果有逐查询、总体和分组指标。`Recall@K` 衡量正例页覆盖比例，`Hit@K` 衡量是否命中至少一页，`RecallAll@K` 衡量是否召回全部标注页。ViDoRe V3 的 nDCG 使用 1／2 级相关性作为增益，其他排序指标将两档都计为正例。MMDocIR 和 DocScope 使用二值页相关性。
- 定位结果包含已召回真值页上的 IoU，以及把未召回的有框真值页计为失败的 `end_to_end_IoU@0.5`。DocScope 的 `fact_coverage@0.5` 在任一有效支持框达标时计该事实被覆盖。它度量的是区域覆盖，分母为可映射到有效证据框的事实。DocScope 的 `RecallAll@K` 仍按所有标注页计，含备选证据页时可结合事实覆盖率解读。
- 列表式请求失败或响应没有可用的排序时，沿用检索序并记录 `api_error`。整批失败时命令以错误退出，诊断结果仍写入产物。正式汇报前应检查失败数、排序来源和输出截断情况。[历史实验报告索引](report/README.md)
