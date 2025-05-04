# Term Project
Code for Term project of CSE576 NLP
## Source code
- `ImageSelection.py`: Code for second step, top-k image selection.<br/>
- `answerGeneration.py`: Code for last step, generate the final answer.<br/>
- `config.yaml`: Store the path to data location.py<br/>
- `datagen.py`: Code for dataloader<br/>
- `main.py`: Main function responsible forrun the pipeline.<br/>
- `metric.py`:  Define functions to calculate accurracy.<br/>
- `mmqa_utils.py`: and`utils.py` Utils for metric.py<br/>
- `openai_metrics.ipynb`: Code to convert the final answer to clean format.<br/>
- `questionDecomposition.py`: Code for first step, decompose the question.<br/>


## Datasets
MultiModalQA Dataset 

MMQA dataset can be downloaded from [mmqa_dev.json](https://drive.google.com/file/d/1hj5c5YPtCt7NzNli18_S76WZJ6XdlTlx/view?usp=sharing)   

Image of MMQA dataset can be downloaded from [images.zip](https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip) 


## How to Run
Modify the config.yaml file to point to the location where you have saved the MMQA data.


To run the pipeline, execute the following command:
```commandline
python main.py
```

