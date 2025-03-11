#!/usr/bin/env python
import argparse

import os
import pandas as pd
from time import time
import json
import torch
from sentence_transformers import SentenceTransformer

def create_embeddings(text, vectorizer, output_path, 
                      output_index, b=500, device="cuda"):
      init_time = time()
      if vectorizer == 'smpnet':
          model = SentenceTransformer('all-mpnet-base-v2', device=device)
      elif vectorizer == 'sgtrt5':
          model = SentenceTransformer('gtr-t5-base', device=device)
      elif vectorizer == 'sdistilroberta':
          model = SentenceTransformer('all-distilroberta-v1', device=device)
      elif vectorizer == 'sminilm':
          model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
      init_time = time() - init_time
     
      vect_time = 0
      with open(output_path, 'w') as o:
          total = len(range(0, len(text), b))
          for i in range(0, len(text), b):
              print(f'\r\t {i//b}/{total}', end='')
              t1 = time()
              temp_text = text[i:i+b]
              temp_index = output_index[i:i+b]
              vectors = model.encode(temp_text)
              t2 = time()
              vect_time += t2-t1
         
              #flushing
              df = pd.DataFrame(vectors)
              df.index = temp_index
              df.to_csv(o, index=True, header=False)
           
      log = {}
      log['init_time'] = init_time   
      log['time'] = vect_time
      log['dimensions'] = vectors.shape[1]
     
      return log
  
def serialize(file, device):
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = "sgtrt5"

    df = pd.read_csv(file, sep=",", index_col=0)
    df = df.fillna('')
    print("\n", file, df.shape)
    
    data = df.apply(lambda x: ' '.join([str(y) for y in x]), axis=1) # Schema-Agnostic format
    text = data.tolist()
    
    file2 = file.replace('.csv', f"_{model}.csv")

    log = create_embeddings(text, model, file2, df.index, device=device)
    return log
