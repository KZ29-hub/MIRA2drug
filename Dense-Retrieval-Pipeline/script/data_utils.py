import json
import jsonlines

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