import json
import random
#load jsonl file


def find_index_evidence(sentences, evidence):
    sentences = sentences.replace('...', '$$').strip()
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
        for i in data:
            k,v = list(i.items())[0]
            id = k
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
    data = load_json('ise-dsc01-warmup.json')
    key_id = data.keys()
    
    train_size = 0.9
    dev_size = 0.1
    
    data_list = [{v : k} for v, k in data.items()]
   
    random.shuffle(data_list)
    train_data = data_list[:int(train_size*len(data_list))]
    dev_data = data_list[int(train_size*len(data_list)):]
    
    
    convertfile(train_data, 'train.jsonl')
    convertfile(dev_data, 'dev.jsonl')
    
