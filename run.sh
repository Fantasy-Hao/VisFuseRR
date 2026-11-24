#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"

# ============================== mmdocir ==============================

# # Qwen3-VL-Embedding-8B
# python -m retriever.encode --dataset-name mmdocir --embedding-model-name Qwen3-VL-Embedding-8B --batch-size 8
# python -m retriever.search --dataset-name mmdocir --embedding-model-name Qwen3-VL-Embedding-8B --fusion sigmoid-zscore --grid-search --topk 50
python -m reranker.pointwise --dataset-name mmdocir --embedding-model-name Qwen3-VL-Embedding-8B --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name mmdocir --embedding-model-name Qwen3-VL-Embedding-8B --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name mmdocir --embedding-model-name Qwen3-VL-Embedding-8B --topk 10

# # jina-embeddings-v4
# python -m retriever.encode --dataset-name mmdocir --embedding-model-name jina-embeddings-v4 --batch-size 8
# python -m retriever.search --dataset-name mmdocir --embedding-model-name jina-embeddings-v4 --fusion sigmoid-zscore --grid-search --topk 50
# python -m reranker.pointwise --dataset-name mmdocir --embedding-model-name jina-embeddings-v4 --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name mmdocir --embedding-model-name jina-embeddings-v4 --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name mmdocir --embedding-model-name jina-embeddings-v4 --topk 10

# # WeMM-Embedding-9B
# python -m retriever.encode --dataset-name mmdocir --embedding-model-name WeMM-Embedding-9B --batch-size 8
# python -m retriever.search --dataset-name mmdocir --embedding-model-name WeMM-Embedding-9B --fusion sigmoid-zscore --grid-search --topk 50
# python -m reranker.pointwise --dataset-name mmdocir --embedding-model-name WeMM-Embedding-9B --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name mmdocir --embedding-model-name WeMM-Embedding-9B --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name mmdocir --embedding-model-name WeMM-Embedding-9B --topk 10

# ============================== docscope ==============================

# # Qwen3-VL-Embedding-8B
# python -m retriever.encode --dataset-name docscope --split test --embedding-model-name Qwen3-VL-Embedding-8B --batch-size 8
# python -m retriever.search --dataset-name docscope --split test --embedding-model-name Qwen3-VL-Embedding-8B --fusion sigmoid-zscore --grid-search --topk 50
python -m reranker.pointwise --dataset-name docscope --split test --embedding-model-name Qwen3-VL-Embedding-8B --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name docscope --split test --embedding-model-name Qwen3-VL-Embedding-8B --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name docscope --split test --embedding-model-name Qwen3-VL-Embedding-8B --topk 10

# # jina-embeddings-v4
# python -m retriever.encode --dataset-name docscope --split test --embedding-model-name jina-embeddings-v4 --batch-size 8
# python -m retriever.search --dataset-name docscope --split test --embedding-model-name jina-embeddings-v4 --fusion sigmoid-zscore --grid-search --topk 50
# python -m reranker.pointwise --dataset-name docscope --split test --embedding-model-name jina-embeddings-v4 --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name docscope --split test --embedding-model-name jina-embeddings-v4 --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name docscope --split test --embedding-model-name jina-embeddings-v4 --topk 10

# # WeMM-Embedding-9B
# python -m retriever.encode --dataset-name docscope --split test --embedding-model-name WeMM-Embedding-9B --batch-size 8
# python -m retriever.search --dataset-name docscope --split test --embedding-model-name WeMM-Embedding-9B --fusion sigmoid-zscore --grid-search --topk 50
# python -m reranker.pointwise --dataset-name docscope --split test --embedding-model-name WeMM-Embedding-9B --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name docscope --split test --embedding-model-name WeMM-Embedding-9B --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name docscope --split test --embedding-model-name WeMM-Embedding-9B --topk 10

# ============================== longdocurl ==============================

# # Qwen3-VL-Embedding-8B
python -m retriever.encode --dataset-name longdocurl --embedding-model-name Qwen3-VL-Embedding-8B --batch-size 8
python -m retriever.search --dataset-name longdocurl --embedding-model-name Qwen3-VL-Embedding-8B --fusion sigmoid-zscore --grid-search --topk 50
python -m reranker.pointwise --dataset-name longdocurl --embedding-model-name Qwen3-VL-Embedding-8B --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name longdocurl --embedding-model-name Qwen3-VL-Embedding-8B --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name longdocurl --embedding-model-name Qwen3-VL-Embedding-8B --topk 10

# # jina-embeddings-v4
# python -m retriever.encode --dataset-name longdocurl --embedding-model-name jina-embeddings-v4 --batch-size 8
# python -m retriever.search --dataset-name longdocurl --embedding-model-name jina-embeddings-v4 --fusion sigmoid-zscore --grid-search --topk 50
# python -m reranker.pointwise --dataset-name longdocurl --embedding-model-name jina-embeddings-v4 --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name longdocurl --embedding-model-name jina-embeddings-v4 --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name longdocurl --embedding-model-name jina-embeddings-v4 --topk 10

# # WeMM-Embedding-9B
# python -m retriever.encode --dataset-name longdocurl --embedding-model-name WeMM-Embedding-9B --batch-size 8
# python -m retriever.search --dataset-name longdocurl --embedding-model-name WeMM-Embedding-9B --fusion sigmoid-zscore --grid-search --topk 50
# python -m reranker.pointwise --dataset-name longdocurl --embedding-model-name WeMM-Embedding-9B --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
# python -m reranker.pointwise --dataset-name longdocurl --embedding-model-name WeMM-Embedding-9B --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
# python -m reranker.listwise --dataset-name longdocurl --embedding-model-name WeMM-Embedding-9B --topk 10
