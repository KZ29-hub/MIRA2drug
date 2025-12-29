import json
import jsonlines
import csv

def readCSV(fp):
    with open(fp, mode='r', encoding='utf-8') as file:
        # 读取文件的所有行
        lines = file.readlines()        
        # 去掉每行的换行符，并分割成列表
        lines = [line.strip() for line in lines]
        # 处理标题行
        headers = lines[0].split(',')
        # 生成字典列表
        data_list = []
        for line in lines[1:]:
            values = line.split(',')
            # 创建字典并添加到列表
            row_dict = {headers[i]: values[i] for i in range(len(headers))}
            data_list.append(row_dict)
    return data_list

def readCSV_desc(fp):
    with open(fp, mode='r', encoding='us-ascii', errors='ignore') as file:
        # 使用 csv.reader 来处理逗号分隔的内容
        reader = csv.reader(file)
        # 将读取的内容转换为列表
        lines = list(reader)

        # 处理标题行
        headers = lines[0][:3]  # 取前3个标题
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

def readCSV_ddi2013(fp):
    with open(fp, mode='r', encoding='us-ascii', errors='ignore') as file:
        # 使用 csv.reader 来处理逗号分隔的内容
        reader = csv.reader(file)
        # 将读取的内容转换为列表
        lines = list(reader)

        # 处理标题行
        headers = lines[0][:12]  # 取前3个标题
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
    
def readJSONL(fp):
    res = []
    with open(fp,"r",encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res  

def writeJSONL(instance,fp):
    with jsonlines.open(fp,'w') as f:
        for sample in instance:
            f.write(sample)

def writeCSV(sample_list, save_path):
    with open(save_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(sample_list[0].keys())

        for item in sample_list:
            writer.writerow(item.values())
    print(f"CSV file {save_path} has been created.")

def writeCSV_xu(sample_list, save_path):
    with open(save_path, mode='a') as file:
        writer = csv.writer(file)
        for item in sample_list:
            writer.writerow(item.values())
    print(f"CSV file {save_path} has been append.")
    
def writeTSV(instance,fp):
    with open(fp,"w") as f:
        for sample in instance:
            line = sample['query_id']+'\t'+sample['query']+'\n'
            f.write(line)

def pad_neg_score(data_dir, split):
    data_split = readJSONL(data_dir/f'{split}.jsonl')
    for i in range(len(data_split)):
        data_split[i]['positives']['score'] = [0.0] * len(data_split[i]['positives']['doc_id'])
        if len(data_split[i]['negatives']['doc_id']) == 0:
            data_split[i]['negatives']['doc_id'] = ['1']
        data_split[i]['negatives']['score'] = [0.0] * len(data_split[i]['negatives']['doc_id'])

    writeJSONL(data_split, data_dir/f'kd_{split}.jsonl')


def rewrite_passage(data_dir):
    fp = data_dir / 'passages.jsonl'
    psg = readJSONL(fp)
    for i in range(len(psg)):
        psg[i]['contents'] =  psg[i]['title'] + '</s></s>' + psg[i]['contents']
    writeJSONL(psg, data_dir/'passages_ss.jsonl')

    psg2 = readJSONL(fp)
    for i in range(len(psg)):
        psg[i]['contents'] =  psg[i]['title'] + ' ' + psg[i]['contents']
    writeJSONL(psg, data_dir/'passages_sp.jsonl')