## Status



## Inspiration for this Project
Tuning is a technique I wanted to practice on and to further build on my AI/ML toolkit. First, during my time at Harvard, it was always on the radar of projects to do but never gotten the chance learn more deeply about it. I took a class on Bioethics at HMS, which I thoroughly enjoyed.Now, after graduation, I have available time on my own for an opportunity to learn about fine-tuning. Second, I as I browsed through many job postings, majority (90%) mentioned that fine-tuning as a required if not desired skill.


## Introduction
This project will focus on two methods of fine-tuning; prompt and prefix. Both are under the umbrella PEFT. 

## Business Case / So What?


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



## Metrics



## Tech Stack



## Workflow




## Results



## Limitations



## Next Steps



# AI/ML End-to-End Build Order