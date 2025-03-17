# Term Project
Code for Human Heterogeneity Invariant Stress Sensing 
## Source code
inference.py Main function that design prompt engineering<br/>
dataloader.py Prepare the dataset as a list of examples<br/>
metric.py Define functions to calculate accurracy or F1 score.<br/>
utils.py Utils for metric.py<br/>



## How to Run
First install ollama and start mistral
```commandline
ollama pull mistral
ollama run mistral
```

To run 
```commandline
python inference.py
```

To select dataset, modify the "dataset_name" variable in inference.py
