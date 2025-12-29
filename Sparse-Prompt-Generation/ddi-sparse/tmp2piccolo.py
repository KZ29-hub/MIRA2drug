import json
import os
import pandas as pd
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
from data_utils import *

def make_arrow(data_list, output_dir):
    features = Features({
        'text': Value('string'),
        'text_pos': Value('string'),
        'text_neg': Sequence(Value('string')),
        'type': Value('string')
    })
    
    # 创建 Arrow 数据集
    arrow_dataset = Dataset.from_dict({key: [d[key] for d in data_list] for key in data_list[0]})
    
    # 保存到磁盘
    os.makedirs(output_dir, exist_ok=True)
    arrow_dataset.save_to_disk(str(output_dir))

def tmp2piccolo(root_path, output_dir):
    train_json = readJSONL(root_path)

    data_list = []

    for example in tqdm(train_json, desc="Processing"):
        text = f"""
        Try to figure out the relation of the drug {example["drug1_name"]} and the drug {example["drug2_name"]}. 
        The smile of {example["drug1_name"]} is {example["drug1_smile"]}, the smile of {example["drug2_name"]} is {example["drug2_smile"]}. 
        The description of {example["drug1_name"]} is {example["drug1_desc"]}, the description of {example["drug2_name"]} is {example["drug2_desc"]}
        """
        text = text.replace("\n", " ")
        text_pos = example["pos"]
        text_neg = [example["neg"]]
        type_ = "cls_contrast"
        
        # Append the data as a dictionary
        data_list.append({"text": text, "text_pos": text_pos, "text_neg": text_neg, "type": type_})

    print(f"----------------------------- 开始写arrow文件 ---------------------------------")
    make_arrow(data_list, output_dir)
    print("OK!!!")

if __name__ == "__main__":
    root_path = "/root/autodl-tmp/dataset_ddi/drugbank_xie_tmp/train.jsonl"
    output_dir = "/root/autodl-tmp/dataset_ddi/piccolo_data/drugbank_xie"
    tmp2piccolo(root_path, output_dir)