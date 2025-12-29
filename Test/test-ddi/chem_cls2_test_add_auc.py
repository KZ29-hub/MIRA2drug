import json
import os
import random
import shutil
from data_utils import *
from tqdm import tqdm
import torch
from sentence_transformers import util,SentenceTransformer
from sentence_transformers.util import semantic_search
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    
def cls_test(root_dir, output_dir, model_dir, model_name):
    drugbank_test_path = os.path.join(root_dir, "test_add.jsonl")
    drugbank_test = readJSONL(drugbank_test_path)
    ## 改1
    ou_drugbank_path = os.path.join(output_dir, "ddi2013_cls2_noadd_test_auc1126.csv")
    model_path = os.path.join(model_dir, model_name)
    
        
    drugbank = []
    # The reference chunk is {example["top1_contents"]}, {example["top2_contents"]}, {example["top3_contents"]}.
    ## 改2
    ## {example["top1_contents"]}
    for example in drugbank_test:
        text = f"""
        Try to figure out the relation of the drug {example["drug1_name"]} and the drug {example["drug2_name"]}. 
        The smile of {example["drug1_name"]} is {example["drug1_smile"]}, the smile of {example["drug2_name"]} is {example["drug2_smile"]}. 
        The description of {example["drug1_name"]} is {example["drug1_desc"]}, the description of {example["drug2_name"]} is {example["drug2_desc"]}.
        """
        drugbank.append(text)


    model = SentenceTransformer(model_path, trust_remote_code=True)
    pool = model.start_multi_process_pool()
    drugbank_embeddings = model.encode_multi_process(drugbank, pool, normalize_embeddings=True, batch_size=100)
    print(drugbank_embeddings.shape)
    print("embedding完成")
    
    corpus = ['negative', 'advise']
    corpus_embeddings = model.encode_multi_process(corpus, pool, normalize_embeddings=True, batch_size=50)
    top_selection = 2
    res_drugbank = semantic_search(drugbank_embeddings, corpus_embeddings, query_chunk_size=100, top_k=top_selection, score_function=util.dot_score)

            
    y_pred = np.array([int(res_drugbank[idx][0]["corpus_id"]) for idx in range(len(drugbank_test))])

    y_true = []
    for example in drugbank_test:
        if example["pos"] == 'negative':
            y_true.append(0)
        else:
            y_true.append(1)
    y_true = np.array(y_true)

    
    # Function to compute softmax
    def softmax(scores):
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        return exp_scores / exp_scores.sum(axis=0)
    
    # Iterate through the data and compute softmax, replacing scores
    y_pred_proba = []
    softmax_data = []
    for sublist in res_drugbank:
        scores = [item['score'] for item in sublist]
        softmax_scores = softmax(np.array(scores))
        
        # Replace scores in the original structure
        for i, item in enumerate(sublist):
            item['score'] = softmax_scores[i]
        softmax_data.append(sublist)
    # print("softmax_data", softmax_data)
    res_drugbank2 = softmax_data

    
    for idx, example in enumerate(drugbank_test):
        if int(res_drugbank2[idx][0]["corpus_id"]) == 0:
               y_pred_proba.append(res_drugbank2[idx][1]["score"])
        else:
               y_pred_proba.append(res_drugbank2[idx][0]["score"])

    
    # ## drugbank
    # cm = confusion_matrix(y_true, y_pred)
    # # 保存为文本文件
    # with open('/root/autodl-tmp/piccolo-embedding/test/test_results/cm.txt', 'w') as f:
    #     for row in cm:
    #         f.write('\t'.join(map(str, row)) + '\n')

    
    drugbank_acc = accuracy_score(y_true, y_pred)
    drugbank_pre = precision_score(y_true, y_pred, average='macro')
    drugbank_recall = recall_score(y_true, y_pred, average='macro')
    drugbank_f1 = f1_score(y_true, y_pred, average='macro')
    # Calculate AUC and AUPR
    drugbank_auc = roc_auc_score(y_true, y_pred_proba)
    drugbank_aupr = average_precision_score(y_true, y_pred_proba)
    
    ### 改3
    new_item_drugbank = {
        "model": 'ddi2013_fold2_noadd1_' + 'stella_' + 'ep10',
        "chem_acc": drugbank_acc,
        "chem_pre": drugbank_pre,
        "chem_recall": drugbank_recall,
        "chem_f1": drugbank_f1,
        "chem_auc": drugbank_auc,
        "chem_aupr": drugbank_aupr
    }
    if os.path.exists(ou_drugbank_path):
        writeCSV_xu([new_item_drugbank], ou_drugbank_path)
    else:
        writeCSV([new_item_drugbank], ou_drugbank_path)
            
        

if __name__ == "__main__":
    ### 改4
    root_dir = "/root/autodl-tmp/dataset_ddi/ddi2013_xie_cls2_tmp/ddi2013_xie_fold2/"
    output_dir = "/root/autodl-tmp/piccolo-embedding/test/test_results/formal_res_auc/"
    ## 改5
    model_dir = "/root/autodl-tmp/piccolo-embedding/scripts/formal/ddi2013_no_add/ddi2013_fold2_cls2_no_add_stella_n1_epoc10_lr1e5_1020/"
    model_names = ["checkpoint-320"]
    
    for model in model_names:
        print(f"-------------------------------- {model} --------------------------------------")
        cls_test(root_dir, output_dir,model_dir, model)
        
    print(f"----------------------------- All models OK!!!!! ----------------------")
         