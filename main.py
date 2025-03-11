import json
import sys
import traceback
import utils.minio_client as mc
import subprocess
from evaluate_responses import calc_scores
from build_prompt import build_prompt
from run_prompt import run_prompt
from serialization import serialize
from blocking import blocking
import os

def run(json):

    try:
        ################################## MINIO INIT #################################
        minio_id = json['minio']['id']
        minio_key = json['minio']['key']
        minio_skey = json['minio'].get('skey', None)
        minio_endpoint = json['minio']['endpoint_url']
        #Init MinIO Client with acquired credentials from tool execution metadata
        mclient = mc.init_client(minio_endpoint, minio_id, minio_key, minio_skey)
        ###############################################################################

        ##### Tool Logic #####
        # First script parameters
        
        log = mc.get_object(json["inputs"]['left_file'][0], 'left_file.csv')
        if 'error' in log:
            raise ValueError(log['error'])
        
        log = mc.get_object(json["inputs"]['right_file'][0], 'right_file.csv')
        if 'error' in log:
            raise ValueError(log['error'])
            
        if 'ground_truth_file' in json["inputs"]:
            log = mc.get_object(json["inputs"]['ground_truth_file'][0], 'gt.csv')
            if 'error' in log:
                raise ValueError(log['error'])
       
        if 'sample_file' not in json["inputs"]: # perform blocking
            if not os.path.exists('gt.csv'):
                raise ValueError('Please provide a Ground Truth File')
        
            device = json["parameters"].get("device", "cpu")
            blocking_k = json['parameters'].get('blocking_k', 10)
            blocking_method = json['parameters'].get('blocking_method', 'intersection')
            log = serialize('left_file.csv', device)
            vect_time = log['time']
            log = serialize('right_file.csv', device)
            vect_time += log['time']
            log = blocking('left_file_sgtrt5.csv', 'right_file_sgtrt5.csv', 'gt.csv',
                           blocking_method, device, blocking_k)
            block_metrics = {'vect_time': vect_time, 'blocking_time': log['blocking_time'],
                            'blocking_precision': log['blocking_precision'],
                            'blocking_recall': log['blocking_recall'],
                            'blocking_f1': log['blocking_f1']}
            if 'sample_file' in json['outputs']:
                mc.put_object(json['outputs']['sample_file'], 'sample_file.csv')
        else:
            log = mc.get_object(json["inputs"]['sample_file'][0], 'sample_file.csv')
            if 'error' in log:
                raise ValueError(log['error'])   
        
        args = {
            "left_file": 'left_file.csv', "right_file": 'right_file.csv',
            "sample_file": 'sample_file.csv', "out_file": "prompts.json" }
        # Optional arguments with defaults
        optional_params = ["seed", "serialization", "task_description", "examples", "experts", "reverse"]
        for param in optional_params:
            if param in json.get("parameters", {}):
                args[param] = json["parameters"][param]
        build_prompt(**args)

        # Second script parameters
        args = {
            "prompts_file": "prompts.json",
            "out_file": "responses.jsonl",
        }
        # Optional parameters
        optional_params = ["model", "log_prob"]
        for param in optional_params:
            if param in json.get("parameters", {}):
                args[param] = json["parameters"][param]
        optional_params = ["endpoint", "token"]
        for param in optional_params:
            if param in json.get("secrets", {}):
                args[param] = json["secrets"][param]                
        run_prompt(**args)
        
        if 'prompts' in json['outputs']:
            mc.put_object(json['outputs']['prompts'], 'prompts.json')
        if 'responses' in json['outputs']:
            mc.put_object(json['outputs']['responses'], 'responses.jsonl')
        
        #Evaluate Responses
        metrics = calc_scores('responses.jsonl', j['parameters']['model'])
        if block_metrics is not None:
            for k,v in block_metrics.items():
                metrics[k] = v

        json= {
                'message': 'Tool Executed Succesfully',
                'output': json['outputs'], 
                'metrics': metrics,
                'status': 200,
              }
        print(json)
        return json

    except Exception as e:
        print(traceback.format_exc())
        return {
            'message': 'An error occurred during data processing.',
            'error': traceback.format_exc(),
            'status': 500
        }
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))