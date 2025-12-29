import json
import os
import random
import shutil
from data_utils import *
from tqdm import tqdm
import torch
from sentence_transformers import util,SentenceTransformer
from sentence_transformers.util import semantic_search
import numpy as np

def neg_padding(root_path, embedding_path, model_path):
    texts = readJSONL(root_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_list = []
    for sample in tqdm(texts, desc="Downloading texts"):
        txt_list.append(sample["chunk"])
    
    model = SentenceTransformer(model_path)
    pool = model.start_multi_process_pool()
    txt_embeddings = model.encode_multi_process(txt_list, pool, normalize_embeddings=True, batch_size=1000)
    print("embedding完成")
    np.save(embedding_path, txt_embeddings)

    print(f"Embeddings已成功保存到{embedding_path}")

if __name__ == "__main__":
    root_path = "/root/autodl-tmp/dataset_retri/merged_data_tmp/merged_pool.jsonl"
    embedding_path = "/root/autodl-tmp/dataset_retri/embedding_pool/embedding_pool.npy"
    model_path = "/root/autodl-tmp/piccolo-embedding_retri/scripts/output_n1s1t1_5053_z5w_d1005/checkpoint-2172/"
    neg_padding(root_path, embedding_path, model_path)