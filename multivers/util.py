"""
Shared utility functions.
"""

import json
import numpy as np
import pathlib
import os
import torch

from transformers import RobertaForMaskedLM
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value = None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())
        return super().forward(hidden_states, 
                               is_index_masked=is_index_masked, 
                               is_index_global_attn=is_index_global_attn, 
                               is_global_attn=is_global_attn,
                               attention_mask=attention_mask, 
                               output_attentions=output_attentions)

class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

def load_jsonl(fname, max_lines=None):
    res = []
    for i, line in enumerate(open(fname)):
        if max_lines is not None and i == max_lines:
            return res
        else:
            res.append(json.loads(line))

    return res


class NPEncoder(json.JSONEncoder):
    "Handles json encoding of Numpy objects."
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NPEncoder, self).default(obj)


def write_jsonl(data, fname):
    with open(fname, "w") as f:
        for line in data:
            print(json.dumps(line, cls=NPEncoder), file=f)


def get_longformer_phobert_checkpoint():
    "https://huggingface.co/bluenguyen/longformer-phobert-base-4096?fbclid=IwAR3ikj_MPxxJPY-Pb261_1J-sG5zTshZkUoNdkxsE2fXClza0M6nn_K7JYI&text=t%C3%B4i+l%C3%A0+%3Cmask%3E"
    
    loaded_model = RobertaLongForMaskedLM.from_pretrained("bluenguyen/longformer-phobert-base-4096")
    checkpoint_model = loaded_model.state_dict()
    return checkpoint_model
    
def get_longformer_science_checkpoint():
    current_dir = pathlib.Path(os.path.realpath(__file__)).parent
    fname = current_dir.parent / "checkpoints/longformer_large_science.ckpt"

    return str(fname)


def unbatch(d, ignore=[]):
    """
    Convert a dict of batched tensors to a list of tensors per entry. Ignore any
    keys in the list.
    """
    ignore = set(ignore)

    to_unbatch = {}
    for k, v in d.items():
        # Skip ignored keys.
        if k in ignore:
            continue
        if isinstance(v, torch.Tensor):
            # Detach and convert tensors to CPU.
            new_v = v.detach().cpu().numpy()
        else:
            new_v = v

        to_unbatch[k] = new_v

    # Make sure all entries have same length.
    lengths = [len(v) for v in to_unbatch.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All values must be of same length.")

    res = []
    for i in range(lengths[0]):
        to_append = {}
        for k, v in to_unbatch.items():
            to_append[k] = v[i]

        res.append(to_append)

    return res


def flatten(z):
    """
    Flatten a nested list.
    """
    return [x for y in z for x in y]


def list_to_dict(xs, keyname):
    return {x[keyname]: x for x in xs}
