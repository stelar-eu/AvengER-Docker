import re
import os
import json
import pandas as pd

def calc_recall(true, preds):
    return len(true & preds) / len(true)

def calc_precision(true, preds):
    return len(true & preds) / len(preds)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate(true, preds):
    recall = calc_recall(true, preds)
    precision = calc_precision(true, preds)
    f1 = f1_score(precision, recall)
    return recall, precision, f1

def find_integers(input_string):
    # Use regular expression to find all integers in the string
    integers = re.findall(r'-?\b\d+\b', input_string)
    # Convert the found integers from strings to integers
    integers = [int(num) for num in integers]
    return integers


def find_integers_in_brackets(text):
    # Use regex to find all numbers enclosed in square brackets
    integers = re.findall(r'\[(\d+)\]', text)
    # Convert the found integers from strings to integers
    integers = [int(num) for num in integers]
    return integers

def find_integers_in_brackets_orca(text):
    # Use regex to find all numbers enclosed in square brackets
    # pattern = r'### Final answer: \[(\d+)\]'
    pattern = r'### Final answer:\s*(?:\{[^}]+\}\s*)?\[(\d+)\]'
    integers = re.findall(pattern, text)
    # Convert the found integers from strings to integers
    integers = [int(num) for num in integers]
    if len(integers) == 0: #Most likely answered None or paraphrased this one.
        integers = [0]
        # print(text)
    return integers

def find_integer(response, model):
    if 'llama' in model:
        t_preds = find_integers_in_brackets(response)
    elif 'mistral' in model:
        t_preds = find_integers_in_brackets(response)
        if len(t_preds)==0:
            t_preds = find_integers(response) 
    elif 'orca' in model:
        t_preds = find_integers_in_brackets_orca(response)
    else:
        t_preds = find_integers_in_brackets(response)
        if len(t_preds)==0:
            t_preds = find_integers(response)
    return t_preds


def calc_scores(file, model):
    true, preds = [], []
    total_time = 0
    with open(file) as f:
        j = json.load(f)
        
        total_size = 0
        for line in j['responses']:
            
            t_preds = find_integer(line['response'], model)
                
            if len(t_preds)==0:
                pred = -1 # llm gave no prediction
            elif len(t_preds) == 1:
                pred = t_preds[0] # only one prediction, desired behavior
            else:
                pred = t_preds[-1] # probably the last number is the predicted one
            
            # if pred > 0 and pred - 1 >= len(j['options']): # error in prediction
            #     pred = -1
            
            if line['ground_truth'] != -1:    
                true.append((line['query_id'], line['answer']))
            
            if pred > 0: # not "error or None of the above" #TODO
                preds.append((line['query_id'], pred))
            total_time += line['time']
            total_size += len(line['response'])
        total_size /= len(j['responses'])
    
    true = set(true)
    # true = set(ground_results) #TODO: Original ground_truth
    preds = set(preds)
    
    recall, precision, f1 = evaluate(true, preds)
    
    # log = {k:v for k,v in j['settings'].items() if k not in ['in_dir', 'logdir']}
    log = {}
    log['recall'] = recall
    log['precision'] = precision
    log['f1'] = f1
    log['time'] = total_time
    log['size'] = total_size

    return log        

