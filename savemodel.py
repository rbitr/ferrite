import sys
import struct
import json
import torch
import numpy as np

#from transformers import AutoModel, AutoTokenizer 
from sentence_transformers import SentenceTransformer
import re

if len(sys.argv) > 1:
    dir_model = sys.argv[1]
else:
    dir_model = "msmarco-distilbert-base-dot-prod-v3"

with open(dir_model + "/tokenizer.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

with open(dir_model + "/modules.json", "r", encoding="utf-8") as f:
    modules = json.load(f)

st_model =  SentenceTransformer(dir_model)

list_vars = st_model[0].state_dict() # transformer

def strip(x: str):
    x = "auto_model." + x
    print(x)
    y = list_vars[x]
    assert y.view(-1)[0].dtype == torch.float32
    return y.numpy()

outfile = sys.argv[2] if len(sys.argv) > 2 else "msmarco-distilbert-base-dot-prod-v3_converted_full.bin"

with open(outfile, mode='wb') as of:
    header_format = 'iiiiiii'
    header_values = [
        hparams['dim'], hparams['hidden_dim'], hparams['n_layers'],
        hparams['n_heads'], 0, len(encoder['model']['vocab']),
        hparams['max_position_embeddings']
    ]
    header = struct.pack(header_format, *header_values)
    of.write(header)

    layer_names = [
        'embeddings.word_embeddings.weight',
        'embeddings.position_embeddings.weight',
        'embeddings.LayerNorm.weight',
        'embeddings.LayerNorm.bias'
    ]

    for l in range(hparams['n_layers']):
        layer_names.extend([
            f'transformer.layer.{l}.attention.q_lin.weight',
            f'transformer.layer.{l}.attention.q_lin.bias',
            f'transformer.layer.{l}.attention.k_lin.weight',
            f'transformer.layer.{l}.attention.k_lin.bias',
            f'transformer.layer.{l}.attention.v_lin.weight',
            f'transformer.layer.{l}.attention.v_lin.bias',
            f'transformer.layer.{l}.attention.out_lin.weight',
            f'transformer.layer.{l}.attention.out_lin.bias',
            f'transformer.layer.{l}.sa_layer_norm.weight',
            f'transformer.layer.{l}.sa_layer_norm.bias',
            f'transformer.layer.{l}.ffn.lin1.weight',
            f'transformer.layer.{l}.ffn.lin1.bias',
            f'transformer.layer.{l}.ffn.lin2.weight',
            f'transformer.layer.{l}.ffn.lin2.bias',
            f'transformer.layer.{l}.output_layer_norm.weight',
            f'transformer.layer.{l}.output_layer_norm.bias'
        ])

    for name in layer_names:
        w = strip(name)
        of.write(memoryview(w))

    # Linear weights at the end
    print("linear.weight")
    y = st_model[2].state_dict()['linear.weight']
    assert y.view(-1)[0].dtype == torch.float32
    of.write(memoryview(y.numpy()))


vname = sys.argv[3] if len(sys.argv) > 3 else "tokenizer.bin"

vocab = encoder["model"]["vocab"]
# write out vocab
max_len = max([len(bytes(v,"utf-8")) for v in vocab])
print("Maximum word size: ", max_len)
with open(vname, "wb") as f:
    f.write(struct.pack("i", max_len))

    for v in vocab:
        vb = bytes(v,"utf-8")
        f.write(struct.pack("ii", 0, len(vb)))
        f.write(struct.pack(f"{len(vb)}s",vb))
