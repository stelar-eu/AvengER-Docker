import os
from time import time
import argparse
import json
# import ollama
import traceback
# from run_prompt_log import MatchingSQ
import pandas as pd
# from ollama import Client
from openai import OpenAI

def run_prompt(
    prompts_file: str,
    out_file: str,
    model: str,
    log_prob: bool = False,
    endpoint: str = "http://localhost:11434",
    token: str = "ollama",
    
):
    """
    Processes prompts using a specified language model (LLM) and saves the output.

    Parameters:
    ----------
    prompts_file : str
        Path to the file containing the prompts to be processed.
    out_file : str
        Path to the file where the results will be saved.
    model : str
        Path to the language model to use for processing.
    log_prob : bool, optional, default=False
        Whether to include log probabilities in the results.
    endpoint : str, optional, default='http://localhost:11434'
        Endpoint URL for the language model server.
    token : str, optional, default='ollama'
        Authentication token for accessing the endpoint.

    Returns:
    -------
    None
        The function processes the prompts and writes the results to the output file.
    """
    
    with open(prompts_file) as f:
        j = json.load(f)
        
    responses = []
    
    # if log_prob:
    #     msq = MatchingSQ(model)
    
    for no, prompt in enumerate(j['prompts']):
        if no >=5:
            break
        if no % 10 == 0:
            print('Query {}/{}\r'.format(no, len(j['prompts'])), end='')
            
        # if prompt['query_id'] not in unexamined: #TODO
        #     continue
        # if len(prompt['prompt']) >= q_99:
        #     missed += 1
        #     continue

        query_time = time()
        
        if log_prob:
            # response = msq.rank(prompt['prompt'], keep_neg=True)
            # response = msq.rank(prompt['prompt'])
            # query_time = time() - query_time
            # log = {'dataset': dataset, 'query_id': prompt['query_id'],
            #        'ground_truth': prompt['ground_truth'], 'response': response,
            #        'time': query_time
            #        }
            # responses.append(log)
            pass
        else:
            if '32B' in model:
                options = {"temperature":0, 'num_predict': 64}
            else:
                options = {"temperature":0}
                
            try:
                
                client = OpenAI(base_url=endpoint, api_key=token)
                response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt['prompt']}])
                response = response.choices[0].message.content

                # import ollama
                # response = ollama.generate(model=model, prompt=prompt['prompt'],
                #                            options=options)
                # client = Client(host=endpoint)
                # response = client.generate(model=model, prompt=prompt['prompt'],
                #                            options=options)
                # response = response['response']
                
            except:
                traceback.print_exc() 
                continue                
            query_time = time() - query_time
            
            log = {'query_id': prompt['query_id'], 'ground_truth': prompt['ground_truth'], 
                    'answer': prompt['answer'], 'response': response, 'time': query_time
                   }
            responses.append(log)

    # print('\nMissed: ', missed)
    j['settings']['model'] = model
    path2 = out_file
    # os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        logs = {'settings': j['settings'], 'responses': responses}
        f.write(json.dumps(logs, indent=4))
    
    # if log_prob:
    #     total = []
    #     for r in responses:
    #         for x in r['response']:
    #             total.append((r['query_id'], x[0], x[1], r['ground_truth']))
    #     total = pd.DataFrame(total, columns=['D1','D2','Score','True'])
    #     path2 = j['settings']['sample_dir'].replace('PT', 'COMEM') + j['settings']['dataset'] + "_sample.csv"
    #     os.makedirs(os.path.dirname(path2), exist_ok=True)
    #     total.to_csv(path2, header=True, index=False)
