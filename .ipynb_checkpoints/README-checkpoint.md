# Parameter-Efficient Fine-Tuning (Prompt and Prefix Tuning)
> Fine-Tuned with Prompt and Prefix adapters; both reached **≥0.71 accuracy** and **≥0.64 macro F1** on the test set (Base 0.45 / 0.39), with **no overfitting observed** as train and eval losses decreased and **eval stabilized around epochs 3 to 5**.


## Inspiration
I wanted learned more about fine-tuning and I decided to try something different other than LoRa. I found these two techniques online from this [YouTube video by Dr. Maryam Miradi](https://www.youtube.com/watch?v=HkZOGGvZzg4&t=422s) where she brielfy went over the concept. I then decided to embark to learn more and apply it to my personal interests. 


## Introduction
Full fine-tuning sounds powerful but is rarely the right first move in healthcare: it needs large labeled datasets, long training cycles, expensive GPUs, and strict MLOps to track new weights. There needs to be better options:

1) **Prompt tuning** is lightweight adaptation that keeps the base model frozen. Prompt tuning changes only the instructions and response format, in other words, no weights updated, so teams can A/B test quickly, ship improvements in hours, and revert instantly if needed.
  
2) **Prefix tuning (PEFT**) learns tiny trainable “prefix” vectors while leaving the foundation model untouched, delivering durable accuracy gains with a small compute budget and simple change control. Together, these approaches cut time-to-value, reduce spend, and make it easy to tailor a single 7B base to many biomedical QA workflows without re-platforming.

These two fine-tuning options matters to the business because it delivers **speed to value**, letting teams iterate in hours or days instead of weeks, while **lowering cost** by training and storing small adapters and reusing the same base model across use cases. Keeping foundation weights immutable **strengthens governance and risk with easier audits**, rollbacks, and approvals. It also increases operational flexibility by enabling different adapters for different clinics or workflows without re-platforming.

## Dataset
I used the **PubMedQA-Labeled** from the [pubmedqa github](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json) from `yes`/`no` subset (I dropped the `maybe` category). Each record has a `question`, `supporting contexts`, and a `final_decision` label. In terms of cleaning, I normalized labels to lowercase yes/no, stripped whitespace, dropped rows with empty text, and deduplicate by (question, contexts) to avoid leakage. I then stratified an 80/10/10 split with a fixed seed (42) so the class ratio (~62% yes / ~38% no) is preserved across train/val/test.

| Split     |       N |     Yes |      No |     Yes % |      No % |
| --------- | ------: | ------: | ------: | --------: | --------: |
| Train     |     712 |     442 |     270 |     62.1% |     37.9% |
| Val       |      89 |      55 |      34 |     61.8% |     38.2% |
| Test      |      89 |      55 |      34 |     61.8% |     38.2% |
| **Total** | **890** | **552** | **338** | **62.0%** | **38.0%** |


## Model
**`google/flan-t5-xl`** from [HuggingFace](https://huggingface.co/google/flan-t5-xl) is a general-purpose, instruction-tuned T5 model using an encoder–decoder (seq2seq) architecture. Trained on a wide mix of tasks, it follows prompts well and maps text-to-text inputs (e.g., question + context → short answer). Its versatility and promptability make it an ideal backbone for parameter-efficient fine-tuning (Prompt/Prefix Tuning) while keeping the base model frozen.

## Workflow
```text
Dataset (PubMedQA yes/no; train/val/test CSV)
  |
  ├─ Normalize labels; schema check; apply prompt template
  └─ Fixed seed (given splits)
  |
Training (FLAN-T5-XL frozen + PEFT)
  ├─ Prompt Tuning (virtual tokens from config)
  └─ Prefix Tuning (virtual tokens from config)
  |
Evaluation (same template on val/test)
  ├─ Greedy generate (max_new_tokens=2), normalize to {yes,no}
  └─ Write preds_{split}.csv, metrics_{split}.json, cm_{split}.csv
  |
Stats (paired on test)
  ├─ McNemar (ΔAccuracy, n01/n10, p)
  └─ Paired bootstrap (ΔAccuracy, ΔMacro-F1, 95% CI)
  |
Outputs
  ├─ Plots: performance_bar_test, latency_bar_test, param_footprint_bar,
  │         eval_*_per_epoch, cm_*
  └─ Table: performance + significance (from stats_eval_test.csv)
```

## Metrics
**Macro-F1**: the average of the F1 scores computed separately for each class, weighting classes equally. Useful for imbalanced data because minority classes count as much as majority ones.

**Accuracy**: the proportion of predictions that are correct. Can be misleading with class imbalance since always predicting the majority class can look good.

**F1 (No)**: the harmonic mean of precision and recall for the “no” class only. Rewards finding “no” cases while avoiding false “no” predictions.

**F1 (Yes)**: the harmonic mean of precision and recall for the “yes” class only. Rewards finding “yes” cases while avoiding false “yes” predictions.

**Latency mean / p95 (s)**: Mean is the average wall-clock time per example in seconds. p95 is the 95th-percentile latency, showing tail slowdowns.

**Trainable Params (M)**: number of parameters updated during fine-tuning, in millions. Indicates adapter size and the memory/compute needed to train and store the tuned weights.


## Results: Table, Performance

| Method        |  Macro-F1 |  Accuracy | F1 (No) | F1 (Yes) | Latency mean / p95 (s) | Trainable Params (M) |
| ------------- | --------: | --------: | ------: | -------: | ---------------------: | -------------------: |
| Base          | **0.389** | **0.449** |   0.581 |    0.197 |          0.016 / 0.016 |            **0.000** |
| Prompt Tuning | **0.648** | **0.719** |   0.490 |    0.806 |          0.021 / 0.021 |            **0.205** |
| Prefix Tuning | **0.703** | **0.742** |   0.596 |    0.810 |          0.017 / 0.017 |            **4.915** |


Both adapters deliver large gains over Base (Accuracy 0.449 → 0.719–0.742, Macro-F1 0.389 → 0.648–0.703). The biggest jump is on the “yes” class (F1: 0.197 → 0.81), while “no” stays similar (Prefix 0.596, Prompt 0.49, Base 0.581). Prefix Tuning is the top performer (Acc 0.742, Macro-F1 0.703) with essentially the same latency as Base (0.017 vs 0.016, respectively). Prompt Tuning is nearly as accurate (Acc 0.719, Macro-F1 0.648) but uses ~0.205M trainable params vs ~4.915M for Prefix—so pick Prompt when memory is tight, and Prefix when you want the best absolute accuracy.


## Results: Visuals
![Macro-F1](./outputs/viz/performance_bar_test.png)

Both adaptations outperform the base where **Prompt** gains **+0.27 Accuracy (0.45→0.72)** and **+0.26 Macro-F1 (0.39→0.65)**, while **Prefix** gains **+0.29 Accuracy (0.45→0.74)** and **+0.31 Macro-F1 (0.39→0.70)**. Prefix tuning is the top performer, edging prompt by +0.02 Accuracy and +0.05 Macro-F1.


![Latency](./outputs/viz/latency_bar_test.png)

Latency is comparable with a range of 0.016–0.02 seconds, so the **quality gains from prompt/prefix tuning come with no material runtime cost**. Prefix essentially matches base, while Prompt adds ~0.004 s from longer inputs.


![param_footprint](./outputs/viz/param_footprint_bar.png)

Train & ship small adapters instead of new models. Adapters are tiny where **Prompt ≈ 204K** and **Prefix ≈ 4.9M** trainable parameters—both <0.1% of a 7B base while keeping foundation weights frozen. Prompt is ~24× smaller (maximally budget-friendly), whereas Prefix spends a few extra million parameters to secure the top accuracy gains.


![prompt](./outputs/viz/cm_prompt_tuning_test.png)

Prompt tuning is highly sensitive to “yes”—it correctly **captures 52/55 positives** (recall ≈ 0.95), which is great for not missing actionable findings. The trade-off is specificity: with 22 false positives vs 12 true negatives (“no” recall ≈ 0.35), it tends to over-call “yes.” For deployment, calibrate the decision threshold and train with more (or harder) no examples or class weighting to cut false positives while preserving the strong yes recall.


![prefix](./outputs/viz/cm_prefix_tuning_test.png)

Prefix tuning shows a balanced error profile: it **captures 49/55 positives** (recall ≈ 0.89) and yields 17 TN / 17 FP on negatives (specificity ≈ 0.50; precision ≈ 0.74). Overall performance is Accuracy ≈ 0.74 (66/89) with Macro-F1 ≈ 0.70, emphasizing strong sensitivity while keeping false alarms moderate—appropriate for clinical QA summaries.


![eval_macro_f1](./outputs/viz/eval_macro_f1_per_epoch.png)

Macro-F1 climbs rapidly in the first **3–5 epochs** and **then plateaus**. Prefix tuning stays ahead by roughly 0.05–0.10 across most epochs, ending around 0.65–0.68, while prompt tuning stabilizes near 0.55–0.57. One exception: prompt spikes at epoch 4 (~0.67) above prefix (~0.60), and epochs 5–7 are roughly tied (~0.63–0.64) before prefix reopens the gap. Variability shrinks after epoch ~5, so longer runs add cost without meaningful gains, so using **early stopping through epoch 4–5 to capture the prompt spike**, then stop at the plateau.


![eval_macro_f1](./outputs/viz/loss_train_vs_eval_four_lines.png)

Train and eval losses fall rapidly and **stabilize by ~3–5 epochs**, then track closely without the eval curve rising—no classic overfitting. Prefix shows a steeper early drop and settles at a slightly lower loss, matching its stronger end metrics. After the plateau, the train–eval gap stays small, indicating stable generalization rather than memorization. **Most value is captured early; extra epochs add cost with little gain.** Thus, adopt early stopping once eval loss flattens for ~2 consecutive epochs.


## Results: Table, Statistical significance vs Base (test)

| Tuned Method  | Test             | Metric   | Effect (Tuned − Base) | 95% CI           |  p-value | Notes                    |
| ------------- | ---------------- | -------- | --------------------: | ---------------- | -------: | ------------------------ |
| Prompt Tuning | Paired bootstrap | Macro-F1 |                +0.258 | [0.1090, 0.4089] |        — | CI≠0; N=89               |
| Prompt Tuning | McNemar (paired) | Accuracy |                +0.270 | —                | 0.004903 | b=46 fixes, c=22 regress |
| Prefix Tuning | Paired bootstrap | Macro-F1 |                +0.316 | [0.1708, 0.4576] |        — | CI≠0; N=89               |
| Prefix Tuning | McNemar (paired) | Accuracy |                +0.292 | —                | 0.001299 | b=44 fixes, c=18 regress |


The table combines two paired evaluations on the same test items (N=89). McNemar’s exact test looks only at disagreements between Base and the tuned model—n01 are fixes (Base wrong, Tuned right) and n10 are regressions (Base right, Tuned wrong)—to ask if accuracy truly improved. The paired bootstrap reports 95% confidence intervals for the effect size (the improvement in Macro-F1 and Accuracy), showing how large and stable the gains are.

Both adapters outperform Base on the paired test set (N=89). **Prompt Tuning**: ΔAccuracy +0.270 (p=0.0049; b=46, c=22) and **ΔMacro-F1 +0.258** [0.1090, 0.4089]. **Prefix Tuning**: ΔAccuracy +0.292 (p=0.0013; b=44, c=18) and **ΔMacro-F1 +0.316** [0.1708, 0.4576]. The bootstrap CIs exclude 0 and McNemar’s discordant counts show many more fixes than regressions, with Prefix providing the larger, more balanced lift.


## Next Steps
- **Balance the dataset**. Find additional “no” examples so the yes/no classes are roughly even; keep a simple source log.
- **Pilot + human review loop**. Run a small clinical pilot with reviewer feedback on model decisions and add a lightweight “escalate/abstain” path for low-confidence cases.

## Conclusion
Lightweight adaptation beats heavy retraining here. On PubMedQA (binary yes/no), both prompt and prefix tuning deliver large gains over base—0.72/0.65 and 0.74/0.70 (Accuracy/Macro-F1) respectively—while keeping latency ~16–20 ms/ex and shipping tiny, governable adapters. Training/eval curves stabilize by ~3–5 epochs with a small gap, indicating a good fit without over- or underfitting. Next, I'll balance classes and validate externally to harden the system for pilot use.

## Tech Stack
Python, PyTroch, Transformers, scikit-learn, pandas, numpy, GPU, PEFT, LLM

### Build order
`config.yaml`  →  `config.py` → `eda.ipynb` (clean and segregate data; found in `data/`) →  `train.py`  →  `eval.py`  → `stats_eval.py`  →  `plots.py`

