import os
import pandas as pd
import argparse
import json


def serialize(row, serialization='DITTO'):
    desc = ''
    if serialization == 'llm':
        return row['description']
        
    for col, val in row.items():
        if pd.isna(val):
            continue
        if serialization == 'DITTO':
            desc += f' [COL] {col} [VAL] {val}'
        elif serialization == 'schema_agnostic':
            desc += f' {val}'    
    return desc


def define_task_description(task_description):
    if task_description == 'SELECT':
        template = """Select a record from the following candidates that refers """ \
                   """to the same real-world entity as the given record. Answer """ \
                   """with the corresponding record number surrounded by \"[]\" """ \
                   """or \"[0]\" if there is none. """ \
                   """\n\nGiven entity record: {} """ \
                   """\nCandidate records:\n""" \
                   """{}\n"""
    elif task_description == 'EXPLAIN':
        template = """Select a record from the following candidates that refers """ \
                   """to the same real-world entity as the given record. Answer """ \
                   """with the corresponding record number surrounded by \"[]\" """ \
                   """or \"[0]\" if there is none and explain briefly why. """ \
                   """\n\nGiven entity record: {} """ \
                   """\nCandidate records:\n""" \
                   """{}\n"""
    elif task_description == 'CONFIDENCE':
        template = """Select a record from the following candidates that refers """ \
                   """to the same real-world entity as the given record. Answer """ \
                   """with the corresponding record number surrounded by \"[]\" """ \
                   """or \"[0]\" if there is none. Accompany your answer by one """\
                   """of the following confidence measures: {{Certain}}, """\
                   """{{Moderately-certain}}, {{Uncertain}}.\n""" \
                   """\n\nGiven entity record: {} """ \
                   """\nCandidate records:\n""" \
                   """{}\n"""
    elif task_description == 'MATCH':
        template = """Do the two entity records refer to the same real-world entity? """\
                   """Answer "Yes" if they do and "No" if they do not. \n"""\
                   """Record 1: {}\nRecord 2: {}"""
    elif task_description == 'COMPARE':
        template = """Which of the following two records is more likely to refer """\
                   """to the same real-world entity as the given record? Answer """\
                   """with the corresponding record identifier "Record A" or """\
                   """ "Record B". \nGiven entity record: {}"""\
                   """\nRecord A: {}\nRecord B: {}"""
    return template


def read_experts(experts_dir, d_dir):
    data = {}
    for model_dir in os.listdir(experts_dir):
        with open(experts_dir + model_dir + "/" + d_dir + ".json") as f:
            j = json.load(f)
            model = j['settings']['model']
            if '/' in model: #path, not model, e.g. in Fine-Tuning
                model = model.split('/')[-2]
            for r in j['responses']:
                qid = r['query_id']
                if type(qid) == str: # for fine-tuning
                    qid = int(qid.split("_")[-1])
                if qid not in data:
                    data[qid] = {}
                data[qid][model] = r['response']
    return data

def shuffle_on_seed(df, seed):
    df = df.sample(frac=1, random_state=seed) # shuffle for position bias
    return df
    
def find_answer_on_shuffle(df, answer):
    if answer is None:
        return None
    
    if answer in df.index:
        position = df.index.get_loc(answer)
        true_position = position+1
    else:
        true_position = 0
    return true_position

def build_example_prompt(file, df1, df2, examples=None, seed=1924, 
                         serialization='DITTO'):
    if examples is None:
        return ""
    
    if examples == "SELECT":
        example_df = pd.read_csv(file)
        example_df.Options = example_df.Options.apply(lambda x: eval(x))
        
        example_prompt = ""
        for index, row in example_df.iterrows():
            if examples == 'SELECT':
                temp_df1 = pd.DataFrame(df1.loc[row['D1']]).T
                temp_df2 = df2.loc[row['Options']]
                temp_df2 = shuffle_on_seed(temp_df2, seed)
                answer = find_answer_on_shuffle(temp_df2, row['True'])
                
                example_prompt += prepare_select_prompt(temp_df1, temp_df2, answer,
                                                        serialization=serialization,
                                                        task_description='SELECT')
        return example_prompt
    elif examples == 'MATCH': #TODO: build example on MATCH prompt
        pass
            

def prepare_description(df, query=False, serialization='DITTO'):
    desc = ''
    for no, (index, row) in enumerate(df.iterrows()):
        if not query:
            desc += f'[{no+1}] '
        desc += serialize(row, serialization) + '\n'
    return desc

def prepare_select_prompt(df1, df2, answer=None, serialization='DITTO', 
                          task_description='SELECT', experts=None):
    
    entity_description = prepare_description(df1, True, serialization=serialization)
    candidate_description = prepare_description(df2, serialization=serialization)
    
    prompt = define_task_description(task_description)
    if task_description == 'EXPERT': #TODO: Add expert (and arguments to function)
        pass
    elif task_description == 'JUSTIFY':
        prompt = prompt.format(entity_description, candidate_description, answer)
    else: 
        prompt = prompt.format(entity_description, candidate_description)

    if task_description == 'EXPERT': #TODO: Add expert answer and not entity answer
        pass        
    elif task_description == 'JUSTIFY':
        pass
    else: 
        if answer is not None: 
            prompt += f"\nANSWER: Record \"[{answer}]\"\n"
            
    if experts is not None:
        prompt += '\nPossible expert answers:\n'
        for no, response in enumerate(experts.values()):
            prompt += f'Expert {no+1}: \"{response}\"\n'

    return prompt

def prepare_match_prompt(df1, df2, answer=None, serialization='DITTO', 
                          task_description='MATCH'): # Add MATCH Prompt
    entity_description = prepare_description(df1, True, serialization=serialization)
    candidate_description = prepare_description(df2, True, serialization=serialization)
        
    prompt = define_task_description(task_description)
    prompt = prompt.format(entity_description, candidate_description)
            
    return prompt

def build_prompt(
    left_file: str,
    right_file: str,
    sample_file: str,
    out_file: str,
    seed: int = 1924,
    serialization: str = "DITTO",
    examples: str = None,
    task_description: str = None,
    experts: str = None,
    reverse: bool = False
):
    """
    Builds a prompt based on the provided input files and configuration options.

    Parameters:
    ----------
    left_file : str
        Path to the left file containing one side of the entity pairs.
    right_file : str
        Path to the right file containing the other side of the entity pairs.
    sample_file : str
        Path to the sample file for example selection or matching.
    out_file : str
        Path to the output file where the constructed prompt will be saved.
    seed : int, optional, default=1924
        Seed for shuffling operations to ensure reproducibility.
    serialization : str
        Serialization strategy to use. Must be one of ['DITTO', 'schema_agnostic', 'llm'].
    examples : str, optional
        Strategy for selecting examples. Can be one of ['SELECT', 'MATCH'].
    task_description : str
        Task description strategy to use. Must be one of ['SELECT', 'EXPLAIN', 'CONFIDENCE', 'MATCH', 'COMPARE'].
    experts : str, optional
        Directory containing expert answers to be included in the prompt.
    reverse : bool, optional
        Whether to add a reverse prompt, applicable for 'MATCH' and 'COMPARE' task descriptions.

    Returns:
    -------
    None
        The function constructs a prompt and saves it to the specified output file.
    """
    df1 = pd.read_csv(left_file, index_col=0)
    df2 = pd.read_csv(right_file, index_col=0)
    
    sample_df = pd.read_csv(sample_file)
    print(df1.shape, df2.shape, sample_df.shape)
    
    # if examples is not None:
    #     file_example = '{}{}_{}_examples.csv'.format(sample_dir, examples, dataset)
    #     example_prompt = build_example_prompt(file_example, df1, df2, examples, 
    #                                           serialization=serialization)
    # else:
    #     example_prompt = ""    
    example_prompt = ""   
        
    # if experts is not None:
    #     expert_answers = read_experts(experts, dataset)
    # else:
    #     expert_answers = None
    expert_answers = None
        
    total_cands = sample_df.groupby('D1')['D2'].apply(list)
    ground_truth = dict(sample_df[['D1', 'True']].drop_duplicates().values)
    
    #TODO: Recall to remove -1 labels!!!!
    logs = []
    lens = []
    for no, (key, cands) in enumerate(total_cands.items()):
        if no % 50 == 0:
            print('Query {}/{}\r'.format(no, len(total_cands)), end='')
        
        temp_df1 = pd.DataFrame(df1.loc[key]).T
        temp_df2 = df2.loc[cands]
        temp_df2 = shuffle_on_seed(temp_df2, seed)
        options = temp_df2.index
        true_answer = int(ground_truth.get(key, -1))
        answer = find_answer_on_shuffle(temp_df2, true_answer)
        if expert_answers is not None:
            experts_dict = expert_answers[key]
        else:
            experts_dict = None
            
        prompt_answer = answer if task_description == 'JUSTIFY' else None
        
        if task_description in ['SELECT', 'EXPLAIN', 'EXPERT', 'CONFIDENCE', 'JUSTIFY']:
            prompt = prepare_select_prompt(temp_df1, temp_df2,
                                           answer=prompt_answer,
                                           serialization=serialization,
                                           task_description=task_description,
                                           experts=experts_dict)
            prompt = example_prompt + prompt
            
            log = {'query_id': key, 'ground_truth': true_answer,
                   'answer': answer, 'options': list(options), 'prompt': prompt,
                   }
            logs.append(log)
            lens.append(len(prompt))
        elif task_description in ['MATCH']:
            prompts = {}
            for index in temp_df2.index:
                temp_temp_df2 = pd.DataFrame(temp_df2.loc[index]).T

                prompt = prepare_match_prompt(temp_df1, temp_temp_df2,
                                              serialization=serialization,
                                              task_description=task_description)
                lens.append(len(prompt))
                
                prompts[index] = [prompt]
                
                if reverse:
                    prompt = prepare_match_prompt(temp_temp_df2, temp_df1,
                                                  serialization=serialization,
                                                  task_description=task_description)
                    lens.append(len(prompt))
                    prompts[index] += [prompt]
                    
            # prompt = example_prompt + prompt
            
            log = {'query_id': key, 'ground_truth': true_answer, 'prompt': prompts}
            logs.append(log)            
            
        elif task_description in ['COMPARE']: #TODO: Add compare
            pass
            
    settings = { "left_file": left_file, "right_file": right_file,
                "sample_file": sample_file, "out_file": out_file,
                "seed": seed, "serialization": serialization, 
                "examples": examples, "task_description": task_description,
                "experts": experts, "reverse": reverse,}
    if len(lens) != 0:
        lens = pd.Series(lens)
        settings['len_q99'] = lens.quantile(0.99)
        settings['len_q50'] = lens.quantile(0.50)
    
    # os.makedirs(out_file, exist_ok=True)
    with open(out_file, 'w') as f:
        logs = {'settings': settings, 'prompts': logs}
        f.write(json.dumps(logs, indent=4))
