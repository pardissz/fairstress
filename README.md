# FairStress

Code for *"Asymmetrically Unfair Decision Making and Reasoning in LLMs"*.

FairStress tests LLM fairness as a decision process: 300 scenarios across 29
domains, 4 degrees of demographic signaling, and 78 adversarial prefixes,
measuring not just accuracy but whether decisions are held and flipped
equally across demographic groups.

## Data

- Full corpus: [`PardisSzah/fairstress`](https://huggingface.co/datasets/PardisSzah/fairstress)
- Release subset (FairStressCore): [`PardisSzah/fairstress-core`](https://huggingface.co/datasets/PardisSzah/fairstress-core)
- Paper: [`FairStress_paper.pdf`](./FairStress_paper.pdf)

## Layout

- `dataset/` — scenario generation and assembly
- `scoring/` — metrics (AccGap, FragGap, TieLean, TieShift)
- `running_models/` — prompting pipelines (answer, reasoning, batch)
- `debiasing/` — 9 debiasing methods on OLMo-3-7B, Gemma-3-12B, Qwen2.5-7B
- `figures/` — figure-generation code and rendered PDFs

Comments/docstrings are stripped. Internal paths are replaced with a `./data`
placeholder — point it at the HF dataset. No credentials included; set
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / a HF token as needed.

## License

MIT, see [`LICENSE`](./LICENSE).
