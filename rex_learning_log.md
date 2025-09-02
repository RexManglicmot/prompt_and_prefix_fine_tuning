Read up on the config.py
SimpleNamespace objects from config.py
For using GPU, need to make sure that the filepaths are not relative and not absolute. Had to do some imports of the files while in the Instance.
`venv_llm`... was corrupt and needed to make venv_2. I tried to save it, but decided to try again. 
Find documentation to figure out how many GBs, etc needed to run experiment generally and for this project
Transformer verison differ from v4 to v5. Had to change some of the args
`nano` is SUPER IMPORTANT when fixing code in an GPU instance, otherwise have to fix outside and try again. Best to do it within the Instance.
Also, if fixing bugs in the Instance make sure to make adjustments in the main script outside of the VM. 
`config.yaml` is really IMPORTANT, it sets up the hyperparameters for the rest of the script. Need to know it inside and out because one change like `context` and `contexts` was an issue while debugging. If make adjustments to the file, make sure the variable names exist in the .py files or if changes are made in the .py files, need to make sure that the corresponding variable is unaffected in the config.yaml file
1X RTX 4090 (24GB) OR 1X RTX 5090 (32GB). 32 GB system RAM and 35+ container disk

Optimizer issue, changed to adamw_torch
lr had to be forced as a float, made changes in train.py
need to understand spacing in nano, the tab key does align correctly
when making the lr into a float, do it outside of the args,
BIG issues downloading data from instance to Mac



Error in issues with the val final_decision classes, says there is 3, but checking EDA, there is only 2 classes; yes and no
Within the eval.py there were "unknown" labels...original ideas was to do a 3-way before, now, it is just binary. had to go through the script and comment out all the unknowns. 
