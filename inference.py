import json
from tqdm import tqdm
import os
from typing import Dict, Any
from collections import Counter
import ollama
from metric import get_accuracy_finqa,get_f1_tatqa
from dataloader import get_data



def create_reader_request_processed(example: Dict[str, Any]):
    
    prompt = 'Read the following text and table, and then answer a question:\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += 'Answer:'
    return prompt


def evaluate(dataset,dataset_name,model_name):
    all_result=[]
    all_example=[]
    for example in tqdm(dataset):
        
        full_prompt = create_reader_request_processed(example)
        
        result = ollama.generate(
                    model=model_name,  # Use an appropriate model
                    prompt=full_prompt,
                    options={
                    "temperature": 0.7,  # Higher temperature for diverse outputs
                    "num_predict": 512,
                    "top_p": 1
                }
            )
        
                
        all_result.append(result)
        all_example.append(example)
        
    
    if dataset_name=='finqa':
        acc=get_accuracy_finqa(all_result,all_example)
        print(f"This approach achieve {acc} Accuracy on FinQA dataset")
    elif dataset_name=='tatqa':
        f1=get_f1_tatqa(all_result,all_example)
        print(f"This approach achieve {f1} F1 score on TatQA dataset")

    
        
    
    

if __name__ == "__main__": 
    dataset_name='tatqa'#finqa,tatqa
    model_name='mistral'
    dataset=get_data(name=dataset_name)
    evaluate(dataset, dataset_name,model_name)
    


