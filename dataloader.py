
from datasets import load_dataset
from datasets import Dataset
import json
Dataset.cleanup_cache_files
def get_data(name):
    if name =='finqa':
        dataset = load_dataset("dreamerdeo/finqa", split="test")
        return [(item["question"], item["answer"], item["pre_text"]+item["post_text"], item["table"]) for item in dataset]
        
    
    elif name =='tatqa':
        with open('tatqa_dataset_test_gold.json', 'r') as file:
            dataset = json.load(file)
            all_data=[]
            for item in dataset:
                table=item['table']['table']
                
                for entry in item['questions']:
                
                    question=entry['question']
                    answer=''
                    if type(entry['answer']) == list:
                        for an in entry['answer']:
                            answer=answer+an+' '
                    else:
                        answer+=str(entry['answer'])
                    
                    if entry['scale']:
                        answer=answer+' '+entry['scale']
                
                    rel_paragraphs=entry['rel_paragraphs']
                    paragraph=''
                   
                    for para in item['paragraphs']:
                        if str(para['order']) in rel_paragraphs:
                            paragraph+=para['text']
                    
                    all_data.append((question,answer,paragraph,table))
            return all_data