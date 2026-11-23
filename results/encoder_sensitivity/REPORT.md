# Encoder sensitivity: frozen transformer embeddings (reviewer request, post-lock)

Embeddings: {
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "torch": "2.11.0+cu128",
  "transformers": "4.46.3",
  "jobs": {
    "thai": {
      "model": "airesearch/wangchanberta-base-att-spm-uncased",
      "articles": 9616,
      "dimension": 768,
      "max_length": 256,
      "seconds": 18.4
    },
    "us": {
      "model": "ProsusAI/finbert",
      "articles": 8080,
      "dimension": 768,
      "max_length": 256,
      "seconds": 4.4
    }
  }
}

| analysis | baseline | relative_improvement | family_ci_low | family_ci_high | holm_p | decision_2pct_family | smallest_margin_met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thai | ZERO | -0.689266 | -0.923334 | -0.49688 | 0.00039996 | W | 0.923334 |
| thai | DIRECT_HUBER | -0.150181 | -0.318079 | -0.000132487 | 0.0291971 | D | 0.318079 |
| thai | ORDINARY_SOFT | -0.26131 | -0.352637 | -0.178861 | 0.00039996 | W | 0.352637 |
| thai | HARD_RDFL | -0.0169712 | -0.0422093 | 0.00633443 | 0.10459 | I | 0.0422093 |
| us | ZERO | -0.388146 | -0.836887 | -0.208411 | 0.00039996 | W | 0.836887 |
| us | DIRECT_HUBER | -0.159499 | -0.292106 | -0.0884656 | 0.00039996 | W | 0.292106 |
| us | ORDINARY_SOFT | -0.202578 | -0.365513 | -0.123254 | 0.00039996 | W | 0.365513 |
| us | HARD_RDFL | -0.0289682 | -0.0656404 | -0.00667963 | 0.00479952 | D | 0.0656404 |

Prediction SD (return units):

| method | thai | us |
| --- | --- | --- |
| DIRECT_HUBER | 0.00703875 | 0.0079814 |
| HARD_RDFL | 0.00812029 | 0.0102776 |
| ORDINARY_SOFT | 0.00592405 | 0.00724869 |
| RDFL_SOFT | 0.00840867 | 0.0105279 |
