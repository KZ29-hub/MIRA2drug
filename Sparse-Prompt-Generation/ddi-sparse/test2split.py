import json
import os
import random
import shutil
from data_utils import *
from tqdm import tqdm
import torch
import csv

def test_split(train_path, test_path, output_dir):
    train_json = readJSONL(train_path)
    test_json = readJSONL(test_path)

    exists = []
    for example in train_json:
        exists.append(example["drug1_id"])
        exists.append(example["drug2_id"])
    exist_set = set(exists)

    no_exist_count = 0
    for example in tqdm(test_json, total=len(test_json), desc="Processing Test..."):
        if example["drug1_id"] not in exist_set or example["drug2_id"] not in exist_set:
            no_exist_count += 1
    print("no_exist_count", no_exist_count)
            
        


if __name__ == "__main__":
    train_path = "/root/autodl-tmp/dataset_ddi/drugbank_xie_tmp/train.jsonl"
    test_path = "/root/autodl-tmp/dataset_ddi/drugbank_xie_tmp/test.jsonl"
    output_path = ""
    test_split(train_path, test_path, output_path)