
from datasets import load_dataset
from datasets import Dataset
import json
Dataset.cleanup_cache_files
def get_data(name):
    dataset=None
    if name =='finqa':
        with open('./data/finqa_test.json') as f:
            dataset = json.load(f)
        
    elif name =='tatqa':
        with open('./data/tatqa_dev.json') as f:
            dataset = json.load(f)
    return dataset
