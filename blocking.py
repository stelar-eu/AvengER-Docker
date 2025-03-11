import pandas as pd
import torch
from time import time

def find_exact_nns(tensor1, tensor2, tensor1_index, tensor2_index, k, device="cuda:0"):
    device = torch.device(device)
    
    tensor11 = torch.Tensor(tensor1).to(device)
    tensor22 = torch.Tensor(tensor2).to(device)
    
    dists = torch.cdist(tensor11, tensor22, p=2)

    topk_dists = torch.topk(dists, k, largest=False)
    results = []
    for no in range(tensor11.size(0)):
        for i in range(k):
            index = topk_dists.indices[no, i].item()
            score = topk_dists.values[no, i].item()
            actual_no = int(tensor1_index[no])
            actual_index = int(tensor2_index[index])
            # if actual_index == 1063 or actual_no == 1063:
                # print(no, index, actual_no, actual_index)
            results.append((actual_no, actual_index, score))
    
    return results

def calc_recall(true, preds):
    return len(true & preds) / len(true)

def calc_precision(true, preds):
    return len(true & preds) / len(preds)

def calc_f1(precision, recall):
    return 2 * precision * recall / (precision+recall)


def blocking(file1, file2, ground_file, method, device, k=10):

    device = torch.device(device if torch.cuda.is_available() else "cpu")    

    df1 = pd.read_csv(file1, header=None, index_col=0)
    df2 = pd.read_csv(file2, header=None, index_col=0)
    print(df1.shape, df2.shape)
    df1_index = df1.index
    df2_index = df2.index
    df1 = torch.Tensor(df1.values)
    df2 = torch.Tensor(df2.values)

    ground_df = pd.read_csv(ground_file, sep=",")
    ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
    
    #query2input
    if method in ['left_to_right', 'union', 'intersection']:
        q2i_time = time()
        q2i_results = find_exact_nns(df1, df2, df1_index, df2_index, k, device)
        q2i_time = time() - q2i_time
        
        if method == 'left_to_right':
            results = q2i_results
            total_time = q2i_time
    
    if method in ['right_to_left', 'union', 'intersection']:
        i2q_time = time()
        i2q_results = find_exact_nns(df2, df1, df2_index, df1_index, k, device) # reverse
        i2q_time = time() - i2q_time
        i2q_results = [(y,x,score) for (x,y,score) in i2q_results]  #reverse 
        
        if method == 'right_to_left':
            results = i2q_results
            total_time = i2q_time
        
    
    if method == 'union':
        union_time = time()
        results_1 = sorted(q2i_results, key=lambda x: (x[0], x[1]))
        results_2 = sorted(i2q_results, key=lambda x: (x[0], x[1]))
        i, j = 0, 0
        union_results = []
        while i < len(results_1) and j < len(results_2):
            if results_1[i][0] < results_2[j][0]:
                union_results.append(results_1[i])
                i += 1
            elif results_1[i][0] == results_2[j][0]:
                if results_1[i][1] < results_2[j][1]:
                    union_results.append(results_1[i])
                    i += 1
                elif results_1[i][1] == results_2[j][1]:
                    union_results.append(results_1[i])
                    i += 1
                else:
                    union_results.append(results_2[j])
                    j += 1
            else:
                union_results.append(results_2[j])
                j += 1            
        # union_results = list(set(i2q_results) | set(q2i_results))
        union_time = time() - union_time
        results = union_results
        total_time = q2i_time + i2q_time + union_time

    if method == 'intersection':
        #intersection
        intersection_time = time()
        # intersection_results = list(set(i2q_results) & set(q2i_results))
        results_1 = sorted(q2i_results, key=lambda x: (x[0], x[1]))
        results_2 = sorted(i2q_results, key=lambda x: (x[0], x[1]))
        i, j = 0, 0
        intersection_results = []
        while i < len(results_1) and j < len(results_2):
            if results_1[i][0] < results_2[j][0]:
                i += 1
            elif results_1[i][0] == results_2[j][0]:
                if results_1[i][1] < results_2[j][1]:
                    i += 1
                elif results_1[i][1] == results_2[j][1]:
                    intersection_results.append(results_1[i])
                    i += 1
                else:
                    j += 1
            else:
                j += 1        
        intersection_time = time() - intersection_time
        results = intersection_results
        total_time = q2i_time + i2q_time + intersection_time
        
    score_results = set([(x,y) for (x,y,_) in results]) 
    recall = calc_recall(ground_results, score_results)
    precision = calc_precision(ground_results, score_results)        
    f1 = calc_f1(precision, recall) 
        
    
    df = pd.DataFrame(results)
    ground_set = {g[0]:g[1] for g in ground_results}
    df[3] = df[0].apply(lambda x: ground_set[x])
    df.columns = ['D1','D2','Score','True']
    # df = df.head(10)
    df.to_csv('sample_file.csv', index=False, header=True)
    
    log = {'blocking_precision': precision, 'blocking_recall': recall, 
           'blocking_f1': f1, 'blocking_time': total_time}
    return log