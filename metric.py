import json
from tqdm import tqdm
import os
from utils import *
from typing import Dict, Any
from collections import Counter

from mmqa_utils import (TEXT_SINGLE_HOP_QUESTION_TYPES, TEXT_AS_FIRST_HOP_QUESTION_TYPES,
                          TEXT_AS_SECOND_HOP_QUESTION_TYPES, TABLE_SINGLE_HOP_QUESTION_TYPES,
                          TABLE_AS_FIRST_HOP_QUESTION_TYPES, TABLE_AS_SECOND_HOP_QUESTION_TYPES,
                          IMAGE_SINGLE_HOP_QUESTION_TYPES, IMAGE_AS_FIRST_HOP_QUESTION_TYPES,
                          IMAGE_AS_SECOND_HOP_QUESTION_TYPES)
from mmqa_utils import list_em,list_f1
import numpy as np


def parse_api_result(result):
    #to_return = []
    text=result['response']
    to_return=[text]
    return to_return



def get_accuracy_finqa(all_result,all_example):
    correct, wrong = 0, 0
    for result,example in zip(all_result,all_example):
        result_counter = Counter()
        codes = parse_api_result(result)
        for r in codes:
            ans = extract_one_num_from_str(r)
            if not ans:
                if 'yes' in r.lower() or 'true' in r.lower():
                    ans = 'yes'
                elif 'no' in r.lower() or 'false' in r.lower():
                    ans = 'no'
            if ans is not None:
                if type(ans) in [dict]:
                    result_counter.update(list(ans.values()))
                elif type(ans) in [list, tuple]:
                    result_counter.update([float(ans[0])])
                elif type(ans) in [str]:
                    result_counter.update([ans])
                else:
                    try:
                        result_counter.update([float(ans)])
                    except Exception:
                        continue
    
        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]        
        else:
            prediction = None
    
        if prediction is None:
            wrong += 1
        elif finqa_equal(prediction, example['answer'], True, True):
            correct += 1
        else:
            wrong += 1
    
        #example.update({'generated': codes, 'executed': prediction})
        #writer.write(json.dumps(example) + '\n')
    return correct / (correct + wrong)


def get_f1_tatqa(all_result,all_example):
    em_and_f1 = TaTQAEmAndF1()
    correct, wrong = 0, 0
    for result,example in zip(all_result,all_example):
    
        answer_counter = Counter()
        units_counter = Counter()
        codes = parse_api_result(result)
        for r in codes:
            ans = r.strip()
            if ans is not None:
                answer_counter.update([str(ans)])
    
        if len(answer_counter) > 0:
            pred_answer = answer_counter.most_common(1)[0][0]
            if pred_answer.startswith('['):
                try:
                    pred_answer = eval(pred_answer)
                except:
                    pred_answer = pred_answer
        else:
            pred_answer = ''
    
        pred_scale = ''
        
        if type(pred_answer) == str:
            pred_answer = [pred_answer]
        if type(pred_answer) == list and type(pred_answer[0]) == str:
            if pred_scale and pred_scale in pred_answer[0]:
                pred_scale = ''
        #print(pred_answer)
        #print(example)
        em_and_f1(ground_truth=example, prediction=pred_answer, pred_scale=pred_scale)
        #example.update({'generated': codes, 'pred_answer': pred_answer, 'pred_scale': pred_scale})
        #writer.write(json.dumps(example) + '\n')
        
        return  em_and_f1.get_overall_metric()[0]
def get_f1_mmqa(all_result,all_example):
    all_f1=[]
    for result,example in zip(all_result,all_example): 
    
        gold_answer = [str(it["answer"]) for it in example["answers"]]
        gold_question_type = example["metadata"]["type"]
        
        pred_result=parse_api_result(result)
  
 
        
        f1=list_f1(pred_result, gold_answer)
        em=list_em(pred_result, gold_answer)
        all_f1.append(f1)
    f1=np.mean(all_f1)
    return f1
        
 
