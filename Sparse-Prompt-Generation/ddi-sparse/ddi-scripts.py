import json
import os
import random
import shutil
from data_utils import *
from tqdm import tqdm
import torch
import csv

def filter_dicts_by_key(data_list, key_value):
    return [d for d in data_list if d.get('DBID') == key_value]
    
def readCSV_ddi2013_xie(fp):
    with open(fp, mode='r', encoding='us-ascii', errors='ignore') as file:
        # 使用 csv.reader 来处理逗号分隔的内容
        reader = csv.reader(file)
        # 将读取的内容转换为列表
        lines = list(reader)

        # 处理标题行
        headers = lines[0][:8]  # 取前3个标题
        print('headers:', headers)

        # 生成字典列表
        data_list = []
        for values in lines[1:]:  # 从第二行开始
            if len(values) < len(headers):
                print(f"Warning: Skipped line due to mismatch in length: {values}")
                continue
            # 创建字典并添加到列表
            row_dict = {headers[i]: values[i] for i in range(len(headers))}
            data_list.append(row_dict)

    return data_list


def drugbank_data_transform(desc_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    ori_train_path = "/root/autodl-tmp/dataset_ddi/western/drug_bank_ddi/Drugbank_po191808_ne191809_rs0_negative_fromRandom Select.csv"
    ori_train = readCSV_ddi2013_xie(ori_train_path)
    descriptions = readCSV_desc(desc_path)

    ou_train_path = os.path.join(output_dir, "train.jsonl")
    ou_test_path = os.path.join(output_dir, "test.jsonl")

    print("drugbank_ori_train", len(ori_train))
    print("description", len(descriptions))

    train_results = []
    for example in tqdm(ori_train, total=len(ori_train), desc="train process:"):
        drug1_dict = filter_dicts_by_key(descriptions, example["DBID1"])
        drug2_dict = filter_dicts_by_key(descriptions, example["DBID2"])
        
        if drug1_dict and drug2_dict:
            drug1_name = drug1_dict[0]["Drugname"]
            drug2_name = drug2_dict[0]["Drugname"]
            description1 = drug1_dict[0]["Description"]
            description2 = drug2_dict[0]["Description"]
    
            if example["label"] == '0':
                drug_pos = 'nagative'
                drug_neg = 'advise'
            else:
                drug_pos = 'advise'
                drug_neg = 'nagative'
    
            ddi_item = {
                "drug1_id": example["DBID1"],
                "drug2_id": example["DBID2"],
                "drug1_name": drug1_name,
                "drug2_name": drug2_name,
                "drug1_smile": example["Drug1"],
                "drug2_smile": example["Drug2"],
                "drug1_desc": description1,
                "drug2_desc": description2,
                "pos": drug_pos,
                "neg": drug_neg
            }
            train_results.append(ddi_item)
    random.shuffle(train_results)
    train_size = 0.99
    train_length = int(len(train_results) * train_size)

    # 切分数据
    train_set = train_results[:train_length]
    test_set = train_results[train_length:]
    # 输出结果
    print("训练集:", len(train_set))
    print("测试集:", len(test_set))
    writeJSONL(train_set, ou_train_path)
    writeJSONL(test_set, ou_test_path)



if __name__ == "__main__":
    desc_path = "/root/autodl-tmp/dataset_ddi/western/drug_description/Drug_description_expand_upload.csv"
    output_dir = "/root/autodl-tmp/dataset_ddi/drugbank_xie_tmp"
    drugbank_data_transform(desc_path, output_dir)