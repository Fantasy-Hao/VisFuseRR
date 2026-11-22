# 报告索引

```
report/
├── README.md
├── overview_20261006.md                   总报告：任务、方法、两阶段主要结果、局限、复现
├── retriever/
│   └── fusion_retrieval_20261006.md       第一部分：图文融合检索与意图自适应加权
└── reranker/
    ├── listwise_rerank_20261007.md        第二部分：listwise 推理重排
    └── pointwise_baseline_20261007.md     第二部分：pointwise 对照基线
```

| 报告 | 内容 | 口径 |
|---|---|---|
| `overview_20261006.md` | 两阶段方法与主要结果汇总 | 第一部分 held-out；第二部分全量 |
| `retriever/fusion_retrieval_20261006.md` | 单路 / 融合指标、权重搜索、held-out 消融、分组最优权重、候选召回 | 1658 query；最终配置为 in-sample，自适应增益为 5 折 held-out（按文档分折） |
| `reranker/listwise_rerank_20261007.md` | listwise 与检索、pointwise 的三方对照，名次位移，分 domain，定位，资源消耗 | 全量 1658 query，Top-10 |
| `reranker/pointwise_baseline_20261007.md` | pointwise 基线指标、名次位移、与检索分融合的对照、重排分分布 | 全量 1658 query，Top-10 |

引用约定：

1. 第一部分的泛化增益以 held-out 结果为准（`fusion_retrieval_20261006.md` §2.3）；最终配置的指标为 in-sample 数值。
2. 第二部分三种系统（检索、pointwise、listwise）使用同一批 query、同一份 Top-10 候选与同一套指标。
3. 各表中的差值由未取整数值计算，可能与表中两数之差相差 0.01 pp。

目录结构、参数与产物说明见 `code/README.md`。
