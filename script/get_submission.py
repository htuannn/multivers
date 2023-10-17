from tqdm import tqdm
import argparse
from pathlib import Path
from underthesea import word_tokenize

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model import MultiVerSModel
from data import get_dataloader, MultiVerSDataset,  Collator
from data_train import get_tokenizer
import util
import torch
import json
import pandas as pd
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--no_nei", action="store_true", help="If given, never predict NEI."
    )
    parser.add_argument(
        "--force_rationale",
        action="store_true",
        help="If given, always predict a rationale for non-NEI.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

def load_json(path_file):
    return json.load(open(path_file, "r", encoding="utf-8"))

def get_predictions(args):
    # Set up model and data.
    model = MultiVerSModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    # If not predicting NEI, set the model label threshold to 0.
    if args.no_nei:
        model.label_threshold = 0.0

    # Since we're not running the training loop, gotta put model on GPU.
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()

    # Grab model hparams and override using new args, when relevant.
    hparams = model.hparams["hparams"]
    del hparams.precision  # Don' use 16-bit precision during evaluation.
    for k, v in vars(args).items():
        if hasattr(hparams, k):
            setattr(hparams, k, v)

    test = load_json("/content/multivers/ise-dsc01-public-test-offcial.json")
    test_dict = [{'id': i,\
    'claim': test[i]['claim'],\
    'context': test[i]['context'],} \
    for i in test.keys()]

    test_df = pd.DataFrame.from_dict(test_dict)
    res = []
    sent_mapping = {}
    for _,v in test_df.iterrows():
        id = v['id']

        claim = v['claim']
        claim = re.sub(r'(\d)\.(\d)', r'\1\2', claim)

        sentences = v['context'].replace('...', '$$').strip()
        sentences = re.sub(r'(\d)\.(\d)', r'\1\2', sentences)
        token = [x.strip() for x in sentences.split('.')]
        sentences = [sentence.replace('$$', '...')+" ." for sentence in token]
        sent_mapping[id]= sentences
        claim_id = id
        abstract_id = -1
        to_tensorize = {"claim": claim,
                        "sentences": sentences,
                        "title": None}
        data_dict = {
            'claim_id': int(id),
            'to_tensorize': to_tensorize,
            'abstract_id': abstract_id}
        res.append(data_dict)

    tokenizer = get_tokenizer(args)
    test_data= MultiVerSDataset(res, tokenizer)
    collator = Collator(tokenizer)
    test_dataloader = DataLoader(test_data,\
                          num_workers=4,\
                          batch_size=16,\
                          collate_fn=collator,\
                          shuffle=False,\
                          pin_memory=True)

    # Make predictions.
    predictions_all = []

    for batch in tqdm(test_dataloader):
        _ ,  preds_batch= model.predict(batch, args.force_rationale)
        predictions_all.extend(preds_batch)

    output={}
    for pred in predictions_all: 
        print(pred.items())
        for k, v in pred.items():
          if v["verdict"] == "NEI":
            e= ""
          else:
            print(v["evidence"][0])
            print(sent_mapping[k])
            e= sent_mapping[k][v["evidence"][0]]
          output.update({str(k):{
                "verdict": v["verdict"],
                "evidence": e}
              })
    return output


def main():
    args = get_args()
    args.encoder_name = "bluenguyen/longformer-phobert-base-4096"
    outname = Path(args.output_file)
    predictions = get_predictions(args)
    print(predictions)

    with open("sample.json", "w") as outfile:
        json.dump(predictions, outfile)

    # Save final predictions as json.
    #formatted = format_predictions(args, predictions)
    
    #util.write_jsonl(formatted, outname)


if __name__ == "__main__":
    main()
