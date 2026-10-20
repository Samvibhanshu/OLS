| System | error before | error after | sleeper-family share of evidence before | after | sleeper posts among cited, after |
|---|---|---|---|---|---|
| Naive RAG + delete exposed posts | 0.1055 ± 0.0109 | 0.0960 ± 0.0109 | 12.3% | 2.7% | 6.4% |
| Naive RAG + delete whole family (oracle) | 0.1055 ± 0.0109 | 0.0945 ± 0.0112 | 12.3% | 0.0% | 0.0% |
| Text dedup + provenance score | 0.0847 ± 0.0178 | 0.0662 ± 0.0189 | 21.6% | 3.2% | 1.6% |
| Lineage-aware RAG (selective rollback) | 0.0853 ± 0.0156 | 0.0670 ± 0.0159 | 22.2% | 3.0% | 0.6% |
