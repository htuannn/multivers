from tqdm import tqdm
import argparse
from pathlib import Path
from underthesea import word_tokenize

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model import MultiVerSModel
from data import get_dataloader, MultiVerSDataset,  Collator
from data_train import get_tokenizer
from split_long_doc import split_long_doc
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

        contexts = v['context']
        if (len(contexts) > 8_000):
            contexts = split_long_doc(doc=contexts)
        else:
            # If context is not split, turn it into a list with a single element (for consistency)
            contexts = [contexts]

        for context in contexts:
            sentences = context.replace('...', '$$').strip()
            sentences = re.sub(r'(\d)\.(\d)', r'\1\2', sentences)
            token = [x.strip() for x in sentences.split('.')]
            sentences = [sentence.replace('$$', '...')+" ." for sentence in token]
            if id in sent_mapping:
                # If item already in the list, add more sentences to it
                sent_mapping[id] = sent_mapping[id] + sentences
            else:
                # If not, create a new item
                sent_mapping[id] = sentences
            claim_id = id
            abstract_id = -1
            to_tensorize = {
                "claim": claim,
                "sentences": sentences,
                "title": None
            }
            data_dict = {
                'claim_id': int(id),
                'to_tensorize': to_tensorize,
                'abstract_id': abstract_id
            }
            res.append(data_dict)

    tokenizer = get_tokenizer(args)
    test_data= MultiVerSDataset(res, tokenizer)
    collator = Collator(tokenizer)
    test_dataloader = DataLoader(test_data,\
                          num_workers=4,\
                          batch_size=args.batch_size,\
                          collate_fn=collator,\
                          shuffle=False,\
                          pin_memory=True)

    # Make predictions.
    predictions_all = []
    predictions_all_probs = []

    for batch in tqdm(test_dataloader):
        preds_prob,  preds_batch= model.predict(batch, force_rationale=True)
        predictions_all_probs.extend(preds_prob)
        predictions_all.extend(preds_batch)

    final_predictions = []
    pred_idx = 0
    while pred_idx < len(predictions_all_probs):
        pred_idx_change = 1 # This is 1 if no split, 2 if split

        pred = predictions_all_probs[pred_idx]
        chosen_pred = pred.copy() # Which prediction assigned to this variable will be added into final predictions list
        chosen_is_second = False # Check whether we use the second prediction or not

        if (pred_idx + 1) < len(predictions_all_probs):
            pred_next = predictions_all_probs[pred_idx + 1].copy()

            if pred['claim_id'] == pred_next['claim_id']:

                if pred['predicted_label'] == 'NEI':
                    chosen_pred = pred_next.copy()
                    chosen_is_second = True

                elif pred_next['predicted_label'] == 'NEI':
                    chosen_pred = pred.copy()

                else:
                    pred_probs = max(pred['label_probs']) * pred['rationale_probs'][pred['predicted_rationale'][0]]
                    pred_probs_next = max(pred_next['label_probs']) * pred_next['rationale_probs'][pred_next['predicted_rationale'][0]]


                    if (pred_probs_next > pred_probs):
                        chosen_pred = pred_next.copy()
                        chosen_is_second = True

                # Skip next prediction since it is used.
                pred_idx_change += 1

        if chosen_is_second:
            # If we use the second prediction as our final prediction,
            # We have to fix index of its sentences (it should be stared
            # from length of the first half of the doc).
            # len(pred['rationale_probs']) is the length of the first half of the doc.
            chosen_pred['predicted_rationale'] = [rationale + len(pred['rationale_probs']) for rationale in chosen_pred['predicted_rationale']]
        
        # Now we change the format of our predictions to be like it was in "predictions_all"
        chosen_pred = {
            str(chosen_pred['claim_id']): {
                'verdict': chosen_pred['predicted_label'],
                'evidence': chosen_pred['predicted_rationale']
            }
        }

        # Save our prediction
        final_predictions.append(chosen_pred)

        # To next prediction
        pred_idx += pred_idx_change

    output={}
    for pred in final_predictions: 
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
