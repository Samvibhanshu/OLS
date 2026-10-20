| method                                 |   cite_organic_root |   cite_distinct_lineage |   cite_campaign |   topic_prec |   eff_n |   n_lineages |
|:---------------------------------------|--------------------:|------------------------:|----------------:|-------------:|--------:|-------------:|
| Naive RAG (top-m)                      |               0.043 |                   0.484 |           0.822 |        0.999 |   8     |        0     |
| Naive RAG (all K)                      |               0.043 |                   0.484 |           0.822 |        0.999 | 300     |        0     |
| MMR diversification (m)                |               0.252 |                   0.867 |           0.508 |        0.999 |   8     |        0     |
| Text near-duplicate collapse (tau=0.8) |               0.117 |                   0.734 |           0.744 |        0.999 | 110.781 |      110.781 |
| Text dedup + provenance score          |               0.893 |                   0.975 |           0.102 |        0.999 |  72.382 |      110.781 |
| Lineage-aware RAG (ours)               |               0.887 |                   0.984 |           0.096 |        0.999 |  66.83  |       92.891 |
| Lineage-aware RAG (oracle lineage)     |               0.896 |                   1     |           0.093 |        0.999 |  62.872 |       91.688 |