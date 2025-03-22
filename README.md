# Term Project
Code for Term project of CSE576 NLP
## Source code
inference.py: Main function responsible for designing prompt engineering.<br/>
dataloader.py: Prepare the dataset as a list of examples<br/>
metric.py: Define functions to calculate accurracy or F1 score.<br/>
utils.py: Utils for metric.py<br/>

## Datasets
FinQA dataset<br/>
TATQA dataset<br/>
MultiModalQA Dataset 

Image of MMQA dataset can be downloaded from [mmqa_dev.json](https://drive.google.com/file/d/1hj5c5YPtCt7NzNli18_S76WZJ6XdlTlx/view?usp=sharing)   

Image of MMQA dataset can be downloaded from [images.zip](https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip) 


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
