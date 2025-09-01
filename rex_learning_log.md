Read up on the config.py
SimpleNamespace objects from config.py
For using GPU, need to make sure that the filepaths are not relative and not absolute. Had to do some imports of the files while in the Instance.
venv_llm... was corrupt and needed to make venv_2. I tried to save it, but decided to try again. 
Find documentation to figure out how many GBs, etc needed to run experiment generally and for this project
Transformer verison differ from v4 to v5. Had to change some of the args
`nano` is SUPER IMPORTANT when fixing code in an GPU instance, otherwise have to fix outside and try again. Best to do it within the Instance.
Also, if fixing bugs in the Instance make sure to make adjustments in the main script outside of the VM. 
`config.yaml` is really IMPORTANT, it sets up the hyperparameters for the rest of the script. Need to know it inside and out because one change like `context` and `contexts` was an issue while debugging. If make adjustments to the file, make sure the variable names exist in the .py files or if changes are made in the .py files, need to make sure that the corresponding variable is unaffected in the config.yaml file

