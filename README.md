## Status
Project is still ongoing with lots of updates and revisions in pursuance.

1) Prepare environment. DONE.

2) Data ingestion & cleaning. DONE.

3) Stratify splits. DONE.

4) config load sanity check. DONE.

5) dataset loader check. DONE.

6) train -- prompt tuning. START HERE.

7) train --prefix tuning

8) Evaluate

9) Plots

10) Qualitative examples

11) optional prompt-engineering baselines

12) optional serverless deployment touchpoints

13) finalize README


## Inspiration for this Project
Tuning is a technique I wanted to practice on and to further build on my AI/ML toolkit. First, during my time at Harvard, it was always on the radar of projects to do but never gotten the chance learn more deeply about it. I took a class on Bioethics at HMS, which I thoroughly enjoyed.Now, after graduation, I have available time on my own for an opportunity to learn about fine-tuning. Second, I as I browsed through many job postings, majority (90%) mentioned that fine-tuning as a required if not desired skill.


## Introduction
Large language models (LLMs) like FLAN-T5 have demonstrated strong capabilities in general-purpose question answering, but their out-of-the-box performance on specialized biomedical tasks is often limited. Traditional fine-tuning of all model parameters is costly, requiring significant GPU resources and long training times—making it impractical for many research teams and organizations.

This project explores parameter-efficient fine-tuning (PEFT) methods, specifically Prompt Tuning and Prefix Tuning, applied to the PubMedQA dataset, a benchmark of biomedical yes/no questions paired with PubMed abstracts. By adapting a compact model (FLAN-T5-small) with less than 0.2% of parameters trainable, we demonstrate how lightweight fine-tuning can substantially improve accuracy, while maintaining efficiency and enabling deployment even on modest hardware such as CPU/M1 laptops.

The workflow includes:
1) Baseline vs PEFT comparisons (Base, Prompt, Prefix)
2) Performance metrics (Accuracy, Macro-F1, Precision/Recall, Confusion Matrix)
3) Efficiency metrics (latency, parameter footprint, training loss curves)
4) Safety metrics (selective prediction with confidence thresholds)
5) Deployment via Hugging Face Serverless Inference for reproducibility and latency benchmarking

This end-to-end pipeline showcases not only the accuracy gains from PEFT but also its practicality for real-world biomedical applications where resources are limited and reliability is critical.

## Business Case / So What?
For organizations in healthcare, biotech, and pharma, the value of this project lies in showing how domain-specific AI systems can be built efficiently and responsibly. By tuning a small fraction of parameters, teams can:

1) Reduce compute costs → no need for large GPU clusters; training can run on commodity hardware.

2) Deploy faster → models can be adapted quickly to new biomedical domains or datasets.

3) Increase reliability → with selective prediction, the model abstains when uncertain, aligning with clinical decision-support requirements.

4) Scale responsibly → models can be shared or hosted via Hugging Face Serverless, avoiding heavy infrastructure investments.

In sum, the benefor of this project is that for an organization, this means faster R&D, reduced operational overhead, and safer AI-assisted decision making, directly supporting innovation in clinical research.

## Tuning Methods
### Prompt Tuning
Prompt tuning does **not** update the model weights, instead learn a set of "virutal tokens" (trainable embeddings). These otkens get prepended to ebery input prompt. The base model stays **frozen** and **only the soft prompt tokens ((e.g. 20k) are trained.** It is a a suite of tunable parameters specified at the starting of the input sequence.

During inference, the model sees
[soft_prompt_tokens] + user_input

and produces task-specific outputs. Prompt tuning nudges the model towards a doman, but odesn not adapt internal layers. 

In sum, prompt tuning does not touch the whole llm model, rather just training on the soft prompts. 

Cons:
Capacity limits: Prompt tuning only learns embeddings; it doesn’t adapt the deeper layers. Dataset size: On very small datasets (like PubMedQA’s 1k labeled Qs), the risk of overfitting is high. You’ll see modest improvements but not miracles.

### Prefix Tuning
Prefix tuning injects learned vectors into **every transformer layer** as "prefixes." The base model stays frozen but again each layer gets a small learned prefix. 


## Optional Zero-Few Shots



## Dataset
The dataset is the Biomedial Question Answering Dataset where it was built on PubMed abstracts. The task is to give a question and a pubmed abstract and allot the model to predict the answer. 

Relevant columns:

1)`id` - anonymous identificaiton number
2)`question`- question regarding article
3)`context` - abstract from PubMed article
4)`label` - binary, yes or no

The dataset was downloaded on the [GitHub repository](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json) in the original JSON format.

### Cleaning /Preprocessing
EDA and cleaning was done in a Jupyter notebook in the `data/` folder.
1) Transposing matrix
2) Checking dtypes and formatting if necessary
2) Dropping irrelevant columns
3) Formating and lower case remaining coliumn names
4) Drop any NA's
5) Drop `maybe` category from the `label` column as it does not pertain to our project

### Split
After cleaning, length was 890 with an uneven split.
Yes: 552
No: 338

Split into train, val, and test with equal with a ratio of ~62% yes and ~38% no. Again, every split has a balanced proportion of yes/no, so the model evaluation is fair.

Train (80% of data)
Yes: 442
No: 270

Val(10% of data)
Yes: 55
No: 40

Test (10% of data)
Yes: 55
No: 34

csvs are contained in the `...data/clean` directory

## Models
`google/flan-t5-small` from HugginFace.

The sizie if ~80M parameters which is light weight and fine for CPU/M1 without GPU. Given hardware contraints, training will be realistic within hours and not extend over to days with potentially bigger models (70B). Further, to train on bigger models would require financial circumstances. 

HAVE IT AS AN OPTION

The model is already instructioned tuned which helps with yes and no questions and small enough such the the PEFT( Prompt/Prefix Tuning) shows clear efficiency gains. 


### Core Experiemnts
Base (frozen, no tuning)
- Use as the baseline model.
- Predicts “yes/no” with no additional training.

Prompt Tuning
- Add ~20 soft tokens to input embedding layer.
- Trainable params ≈ 20k (<0.1%).
- Training: 3 epochs on PubMedQA train split.

Prefix Tuning
-Inject ~5 prefix tokens into each of 18 attention modules (encoder + decoder).
- Trainable params ≈ 200k (~0.2%).
- Same training setup as prompt tuning.

In the end, compare vs base vs prompt vs prefix.

## Metrics (10 total)
### Core
1) Accuracy - % of correct predictions across
 test set.


2) Macro F1 - Average of F1 across “yes” and “no” classes.
Balances precision/recall, ensures minority class (no) is weighted equally.

3) Per-class Precision and Recall
Lets you see if the model is biased toward one label.


4) Confusion Matrix
Visual diagnostic for error patterns.

### Training / PEFT 
5) Training Loss Curve – Cross-entropy loss over steps/epochs.
6) Parameter Footprint – Trainable params vs total model params (%).

### Efficiency
7) Latency (s/req) – Average time per request (Base vs Prompt vs Prefix).

### Safety
8) Coverage – % of examples where confidence ≥ threshold τ.
9) Selective Accuracy – Accuracy on just those “confident” examples.
10) Coverage–Accuracy Curve – Sweep τ values to show trade-off.


## Core Ploits
1) Performance Bar Chart (Accuracy & Macro-F1)
- Bars for Base, Prompt Tuning, Prefix Tuning.
- Metric link: (1) Accuracy, (2) Macro F1.

2) Parameter Footprint Bar Chart

- Shows total params vs trainable params for Prompt & Prefix.
- Metric link: (6) Parameter Footprint.

3) Training Loss Curve
- Loss vs steps/epochs for Prompt & Prefix.
- Metric link: (5) Training Loss.

4) Latency Bar Chart
- Mean s/req for Base, Prompt, Prefix (optionally Few-shot).
- Metric link: (7) Latency.


5) Confusion Matrix Heatmaps
- Binary 2×2 heatmaps (Yes/No) per method.
- Metric link: (4) Confusion Matrix.

6) Coverage–Accuracy Curve (Selective Prediction)
- Line(s) for Prompt/Prefix, optional Base; shows accuracy vs coverage as τ varies.
- Metric link: (8) Coverage, (9) Selective Accuracy, (10) Curve.

Optional Plots (if baselines enabled)

7) Performance Bars Extended
- Adds Zero-shot and Few-shot (k=3,5) alongside Base, Prompt, Prefix.
- Expands plot #1.

8) Latency Bars Extended
- Adds Zero-/Few-shot to latency comparison.
- Expands plot #4.

## Tech Stack
Python
PyTorch
HuggingFace Transformers
PEFT
Pandas
Sciket-Learn
Matplotlib
PyYAML
git

## Workflow



## Results



## Limitations



## Next Steps
1) Train with a bigger model. A larger model will be trained on more parameters and it encode richer language and sees more resoning patterns. For example, the PedMedQA dataset is very niche, by using a migger model, it has already captued more biomedical resoning in their pretrainiing so the PEFT fidnds it easier to adapt to them with very little data. 


# AI/ML End-to-End Build Order



Vas.ai setup

1) Set up account
2) Put $25 into account
3) Choose template, I chose Pytorch as it was suitable to my build
4) Choose the closest GPU cloud server available
5) Create Instance
6) Find SSH key 