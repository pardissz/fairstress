# Third-Party Dependencies

This repository contains no vendored or copy-pasted third-party source code —
every debiasing method (LEACE, INLP, CAA, CAST, LoReFT, Shifting Perspectives,
etc.) is an original implementation. The only external IP involved is the
pip packages listed in `requirements.txt`, all of which are permissively
licensed.

| Library | License | Used for |
|---|---|---|
| numpy | BSD-3-Clause | Numerical arrays |
| pandas | BSD-3-Clause | Tabular data / metric aggregation |
| scipy | BSD-3-Clause | Statistics (bootstrap CIs, etc.) |
| scikit-learn | BSD-3-Clause | INLP projections, calibration metrics |
| matplotlib | PSF-based (OSI-approved) | Figure generation |
| seaborn | BSD-3-Clause | Figure generation |
| torch (PyTorch) | BSD-3-Clause (modified) | Model loading, activation hooks (CAA/CAST/LoReFT) |
| transformers | Apache-2.0 | HF model loading |
| vllm | Apache-2.0 | Batched inference |
| datasets | Apache-2.0 | Loading the FairStress splits |
| huggingface_hub | Apache-2.0 | Hugging Face Hub upload/download |
| openai | Apache-2.0 | GPT-4o / GPT-4o-mini API calls (scenario generation, judging) |
| anthropic | MIT | Claude API calls |
| concept-erasure | MIT | LEACE implementation |
| python-dotenv | BSD-3-Clause | Loading API keys from a local `.env` (never committed) |

**Summary:** all 15 dependencies are MIT, BSD-3-Clause, or Apache-2.0 —
no copyleft (GPL/AGPL/LGPL) licenses are present. Code itself is released
under MIT (see `LICENSE`).
