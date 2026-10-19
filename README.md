![FairStress: Asymmetrically Unfair Decision Making and Reasoning in LLMs](assets/fairstress_banner.png)

# FairStress

Code for the paper **"Asymmetrically Unfair Decision Making and Reasoning in LLMs"**.

Pardis Sadat Zahraei, Faeze Brahman, Dilek Hakkani-Tür, Gokhan Tur, Maarten Sap
Allen Institute for AI (Ai2) · University of Illinois Urbana-Champaign · Carnegie Mellon University

📄 Paper: *link coming soon* · 🤗 Dataset: [`PardisSzah/fairstress`](https://huggingface.co/datasets/PardisSzah/fairstress) · ⚖️ License: MIT (code), CC BY 4.0 (data)

FairStress tests LLM fairness as a decision process: 300 two-candidate decisions across 29 high-stakes
domains, 4 degrees of demographic signaling, 33 minority/majority contrasts and 78 adversarial prefixes.
It measures not just whether a model's answer is correct, but whether it holds a correct decision, and a
tied decision, equally firmly regardless of which group it favors.

![The FairStress framework](figures/png/fig1_framework.png)

## What we find

- With explicit identity, accuracy differs by only 2.9 points between groups, yet one added sentence makes
  15 models abandon correct answers favoring the majority 9.3 points more often, and flip ties toward the
  minority 23.9 points more often.
- Nine post-hoc debiasing methods do not reliably reduce this asymmetry.
- Models justify majority-coded choices by merit and minority-coded choices by identity.
- OLMo-3's training data frames demographic groups asymmetrically and contains almost no examples of a
  merit-based decision.

## Data

The dataset is on the Hugging Face Hub with two splits:

| Split | Items | Use |
|---|---|---|
| `full` | 13,032,104 | The complete corpus behind the paper's results. |
| `core` | 50,012 | **FairStress-Core**, a subset selected with item response theory; a cheaper option. |

```python
from datasets import load_dataset

core = load_dataset("PardisSzah/fairstress", split="core")
core.to_json("data/fairstress_core.jsonl")   # the runners and the scorer read JSON lines
```

## Installation

```bash
pip install -r requirements.txt
```

API models need `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment (or in a local `.env`).

## Quick start

Run a model on FairStress-Core and score it:

```bash
# 1. responses (vllm for open models; openai, anthropic or openai-compat for APIs)
python running_models/run.py \
  --benchmark data/fairstress_core.jsonl \
  --backend vllm --model allenai/OLMo-3-7B-Instruct \
  --output outputs/responses_olmo3_7b.jsonl

# 2. metrics, with bootstrap confidence intervals
python scoring/score.py \
  --benchmark data/fairstress_core.jsonl \
  --responses outputs/responses_olmo3_7b.jsonl \
  --output outputs/scores_olmo3_7b.json
```

`running_models/run_reasoning.py` collects chain-of-thought and thinking traces, and
`running_models/run_batch.py` uses the OpenAI Batch API.

## Metrics

| Metric | Items | Question | Fair value |
|---|---|---|---|
| **Accuracy Gap** | ground truth | Is the model right equally often when the right answer favors either group? | 0 |
| **Robustness Gap** | ground truth + adversarial prefix | When one sentence argues for the weaker candidate, is a correct answer abandoned equally often? | 0 |
| **Tie-Break Lean** | ambiguous (tie) | When neither candidate is better, how often does the model pick the minority candidate? | 50% |
| **Tie-Break Shift** | ambiguous + adversarial prefix | When the model breaks a tie against a candidate, does one sentence arguing for that candidate reverse the choice equally often for both groups? | 0 |

Positive values (above 50% for Tie-Break Lean) mean the model favors the minority-coded candidate
(overcorrection); negative values mean stereotypical behavior. The scorer also reports refusal rates.

## Repository layout

| Folder | Contents |
|---|---|
| `dataset/` | Scenario generation, prefix generation, assembly and validation |
| `running_models/` | Inference for API and open models: answers, reasoning traces, batch jobs |
| `scoring/` | The four metrics, refusal rates and confidence intervals |
| `debiasing/` | The nine post-hoc debiasing methods on OLMo-3-7B, Gemma-3-12B and Qwen2.5-7B |
| `figures/` | Every figure in the paper, with the code and data to redraw it (see `figures/README.md`) |

## Citation

```bibtex
@misc{zahraei2026asymmetrically,
  title  = {Asymmetrically Unfair Decision Making and Reasoning in {LLMs}},
  author = {Zahraei, Pardis Sadat and Brahman, Faeze and Hakkani-T{\"u}r, Dilek and Tur, Gokhan and Sap, Maarten},
  year   = {2026}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The FairStress dataset is released
under CC BY 4.0.

> **Content warning:** the benchmark and the model outputs it elicits include demographic stereotypes
> and may be offensive or upsetting.
