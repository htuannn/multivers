import json
import random
import re
#load jsonl file
from sklearn.model_selection import train_test_split

import pandas as pd

def find_index_evidence(sentences, evidence):
    sentences = sentences.replace('...', '$$').strip()
    sentences = re.sub(r'(\d)\.(\d)', r'\1,\2', sentences)
    token = [x.strip() for x in sentences.split('.')]
    sentences = [sentence.replace('$$', '...') for sentence in token]
    if evidence != None:
        evidence_end = evidence[-1]
        for tk in token:
            if evidence_end == '.':
                tk_temp = tk + '.'
                if tk_temp == evidence:
                    return [token.index(tk)]
            else :
                if tk == evidence:
                    return [token.index(tk)]
    else : 
        return None

def convertfile(data, output):
    label_lookup = {"REFUTED": "REFUTES", "SUPPORTED": "SUPPORTS", "NEI": "NOT ENOUGH INFO"}
    with open(output, 'w', encoding = 'utf-8') as f:
        data = pd.DataFrame.from_dict(data)
        for _,v in data.iterrows():
            id = v['id']

            claim = v['claim']
            label = label_lookup[v['verdict']]

            sentences = v['context'].replace('...', '$$').strip()
            token = [x.strip() for x in sentences.split('.')]
            sentences = [sentence.replace('$$', '...')+" ." for sentence in token]

            evidence_sets = [find_index_evidence(v['context'], v['evidence'])]
            
            if find_index_evidence(v['context'], v['evidence']) == None:
                evidence_sets = []
            negative_sample_id = -1
            abstract_id = -1
            data_dict = {
                'id': int(id),
                'claim': claim,
                'label': label,
                'sentences': sentences,
                'evidence_sets': evidence_sets,
                'negative_sample_id': negative_sample_id,
                'abstract_id': abstract_id
            }

            json.dump(data_dict, f, ensure_ascii=False)
            f.write('\n')
        
def load_json(path_file):
    return json.load(open(path_file, "r", encoding="utf-8"))

if __name__ == '__main__':
    data_dict = load_json('ise-dsc01-train.json')
    train_size = 0.9
    dev_size = 0.1
    
    samples = [{
        'id': i,
        'claim': data_dict[i]['claim'],
        'verdict': data_dict[i]['verdict'],
        'context': data_dict[i]['context'],
        'evidence': data_dict[i]['evidence']
    } for i in data_dict.keys()]

    # Tạo tập huấn luyện và tập kiểm tra với tỷ lệ nhãn giữ nguyên
    X = [sample for sample in samples]
    y = [sample['verdict'] for sample in samples]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=18)

    train_data = {
        'id': [sample['id'] for sample in X_train],
        'claim': [sample['claim'] for sample in X_train],
        'verdict': [sample['verdict'] for sample in X_train],
        'context': [sample['context'] for sample in X_train],
        'evidence': [sample['evidence'] for sample in X_train],
    }

    test_data = {
        'id': [sample['id'] for sample in X_test],
        'claim': [sample['claim'] for sample in X_test],
        'verdict': [sample['verdict'] for sample in X_test],
        'context': [sample['context'] for sample in X_test],
        'evidence': [sample['evidence'] for sample in X_test],
    }

    convertfile(train_data, 'train.jsonl')
    convertfile(test_data, 'dev.jsonl')
    
