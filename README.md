# VisFuseRR

## 在项目根目录执行

```bash
export PYTHONPATH="$PWD/code"
```

## 选择数据集（三选一）

### MMDocIR

```bash
DATASET=mmdocir
```

### DocScope 测试集

```bash
DATASET=docscope
```

### ViDoRe V3 五个英文子集

```bash
DATASET=vidore_v3
```

## 编码与检索

### Qwen3-VL-Embedding-8B

```bash
python3 -m pip install -r requirements-qwen.txt
EMBEDDING=Qwen3-VL-Embedding-8B
python3 -m retriever.encode --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --batch-size 8
python3 -m retriever.search --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --fusion sigmoid-zscore --topk 50 --grid-search
```

### jina-embeddings-v4

```bash
python3 -m pip install -r requirements-jina.txt
EMBEDDING=jina-embeddings-v4
python3 -m retriever.encode --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --batch-size 8
python3 -m retriever.search --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --fusion sigmoid-zscore --topk 50 --grid-search
```

### WeMM-Embedding-9B

```bash
python3 -m pip install -r requirements-wemm.txt
EMBEDDING=WeMM-Embedding-9B
python3 -m retriever.encode --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --batch-size 8
python3 -m retriever.search --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --fusion sigmoid-zscore --topk 50 --grid-search
```

## 重排当前检索结果

### Qwen3-VL-Reranker-8B

```bash
python3 -m pip install -r requirements-qwen.txt
python3 -m reranker.pointwise --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --reranker-model-name Qwen3-VL-Reranker-8B --topk 10 --batch-size 8
```

### jina-reranker-m0

```bash
python3 -m pip install -r requirements-jina.txt
python3 -m reranker.pointwise --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --reranker-model-name jina-reranker-m0 --topk 10 --batch-size 8
```

### Qwen/Qwen3.8-27B

```bash
python3 -m pip install -r requirements-qwen.txt
python3 -m reranker.listwise --dataset-name "$DATASET" --embedding-model-name "$EMBEDDING" --model Qwen/Qwen3.8-27B --topk 10
```
