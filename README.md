
# FairStress

Code for the paper **"Asymmetrically Unfair Decision Making and Reasoning in LLMs"**.

License: MIT (code), CC BY 4.0 (data)

FairStress tests LLM fairness as a decision process: 300 two-candidate decisions across 29 high-stakes
domains, 4 degrees of demographic signaling, 33 minority/majority contrasts and 78 adversarial prefixes.
It measures not just whether a model's answer is correct, but whether it holds a correct decision, and a
tied decision, equally firmly regardless of which group it favors.

![The FairStress framework](figures/png/fig1_framework.png)

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



## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The FairStress dataset is released
under CC BY 4.0.

> **Content warning:** the benchmark and the model outputs it elicits include demographic stereotypes
> and may be offensive or upsetting.
