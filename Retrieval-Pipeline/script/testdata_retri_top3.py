### retrieval 3

import json
import os

from data_utils import *
from tqdm import tqdm
import torch
from sentence_transformers import util,SentenceTransformer
from sentence_transformers.util import semantic_search
import numpy as np

def top_3_retrieval(input_path, embedding_path, txt_path, model_path, ou_path):
    top_selection = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_json = readJSONL(input_path)
    test_query = []
    for example in test_json:
        text = f"""
        Try to figure out the relation of the drug {example["drug1_name"]} and the drug {example["drug2_name"]}. 
        The smile of {example["drug1_name"]} is {example["drug1_smile"]}, the smile of {example["drug2_name"]} is {example["drug2_smile"]}. 
        The description of {example["drug1_name"]} is {example["drug1_desc"]}, the description of {example["drug2_name"]} is {example["drug2_desc"]}
        """
        test_query.append(text)
        
    model = SentenceTransformer(model_path, trust_remote_code=True)
    pool = model.start_multi_process_pool()
    test_embeddings = model.encode_multi_process(test_query, pool, normalize_embeddings=True, batch_size=1500)
    vector_embeddings = np.load(embedding_path)
    print(test_embeddings.shape)
    print(vector_embeddings.shape)
    
    ori_texts = readJSONL(txt_path)

    res = semantic_search(test_embeddings, vector_embeddings, query_chunk_size=100, top_k=top_selection, score_function=util.dot_score)

    ans = []
    for idx, example in tqdm(enumerate(test_json), total=len(test_json), desc="processing.."):
        top1_idx = res[idx][0]["corpus_id"]
        top2_idx = res[idx][1]["corpus_id"]
        top3_idx = res[idx][2]["corpus_id"]
        example["top1_contents"] = {
            "chunk": ori_texts[top1_idx]["chunk"],
            "source": ori_texts[top1_idx]["source"]
        }
        example["top2_contents"] = {
            "chunk": ori_texts[top2_idx]["chunk"],
            "source": ori_texts[top2_idx]["source"]
        }
        example["top3_contents"] = {
            "chunk": ori_texts[top3_idx]["chunk"],
            "source": ori_texts[top3_idx]["source"]
        }

        ans.append(example)
    print(ans[0].keys())
    print("总共", len(ans))
    writeJSONL(ans, ou_path)
    print("添加完成")
    
    
        

if __name__ == "__main__":
    root_path = "/root/autodl-tmp/dataset_ddi/drugbank_twosides_tmp/drugbank_right2/fold0/test.jsonl"
    embedding_path = "/root/autodl-tmp/dataset_retri/embedding_pool/embedding_pool.npy"
    model_path = "/root/autodl-tmp/piccolo-embedding_retri/scripts/output_n1s1t1_5053_z5w_d1005/checkpoint-2172/"
    txt_path = "/root/autodl-tmp/dataset_retri/merged_data_tmp/merged_pool.jsonl"
    ou_path = "/root/autodl-tmp/dataset_ddi/drugbank_twosides_tmp/drugbank_right2/fold0/test_add.jsonl"
    top_3_retrieval(root_path, embedding_path,txt_path, model_path, ou_path)