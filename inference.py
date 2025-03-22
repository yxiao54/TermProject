import json
from tqdm import tqdm
import os
from typing import Dict, Any
from collections import Counter
import ollama
from metric import *
from dataloader import get_data
from mmqa_utils import (TEXT_SINGLE_HOP_QUESTION_TYPES, TEXT_AS_FIRST_HOP_QUESTION_TYPES,
                          TEXT_AS_SECOND_HOP_QUESTION_TYPES, TABLE_SINGLE_HOP_QUESTION_TYPES,
                          TABLE_AS_FIRST_HOP_QUESTION_TYPES, TABLE_AS_SECOND_HOP_QUESTION_TYPES,
                          IMAGE_SINGLE_HOP_QUESTION_TYPES, IMAGE_AS_FIRST_HOP_QUESTION_TYPES,
                          IMAGE_AS_SECOND_HOP_QUESTION_TYPES)


def create_full_prompt_finqa_tatqa(example: Dict[str, Any]):
    
    prompt = 'Read the following text and table, and then answer a question:\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += 'Answer:'
    return prompt

def create_full_prompt_mmqa(example: Dict[str, Any]):
    question_type = example["gold_question_type"]
    prompt = 'Read the following text and table, and then answer a question:\n'

    
    prompt += '\n'.join(example['table_context']) + '\n'
    prompt += '\n'.join(example['text_context'])+ '\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += 'Answer:'
    return prompt

def evaluate(dataset,dataset_name,model_name):
    all_result=[]
    all_example=[]
    for example in tqdm(dataset):
        
        if dataset_name in ['finqa','tatqa']:
            full_prompt = create_full_prompt_finqa_tatqa(example)
        elif dataset_name in ['mmqa']:
            full_prompt = create_full_prompt_mmqa(example)
        else:
            full_prompt=None
      
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
    elif dataset_name=='mmqa':
        f1=get_f1_mmqa(all_result,all_example)
        print(f"This approach achieve {f1} F1 score on mmqa dataset")

    
        
    
    

if __name__ == "__main__": 
    dataset_name='mmqa'#finqa,tatqa,mmqa
    model_name='mistral'
  
    dataset=get_data(name=dataset_name)
    evaluate(dataset, dataset_name,model_name)
    


