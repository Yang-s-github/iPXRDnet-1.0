import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnFunctional
from torch.autograd import Variable
import pickle
from .utils_dist import parse_args
import pdb

from argparse import Namespace
from .parsing import parse_train_args, modify_train_args
from .model import  get_cmpn_encoder,encoder_add_functional_prompt #,chemprop.
from .nn_utils import get_activation_function

from rdkit import RDLogger


torch.cuda.set_device(0)
device = torch.device("cuda")
args = parse_args()

#######################################
## Build Model
#######################################

def build_model_gas_only(tgt,ffn_dim=2048, head=4, dropout=args.dropout, out_both=False):

    model = KANO_gas_only(Generator_KANO(300, tgt, dropout), out_both)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

class KANO_gas_only(nn.Module):
    def __init__(self, generator, out_both):
        super(KANO_gas_only, self).__init__()
        
        args_KANO = parse_train_args()
        modify_train_args(args_KANO)
        args_KANO.cuda=True
        args_KANO.checkpoint_path="data/original_CMPN.pkl"
        self.gas_embed = get_cmpn_encoder(args_KANO)   #shape = 16,500
        encoder_add_functional_prompt(self.gas_embed, args_KANO)
        self.gas_embed.load_state_dict(torch.load(args_KANO.checkpoint_path, map_location='cpu'), strict=False)

        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        
        self.generator = generator
        self.out_both = out_both

    def forward_once(self, gas_list):
        gas_embed_ed = self.gas_embed( 'finetune', False, gas_list, None)
        return gas_embed_ed

    def forward(self, gas_list):
        if self.out_both:
            h = self.forward_once(gas_list)
            return self.generator(h), h

        return self.generator(self.forward_once(gas_list))

class Generator_KANO(nn.Module):
    """Define standard linear + activation generation step."""

    def __init__(self, embed_dim, tgt, dropout):
        super(Generator_KANO, self).__init__()
        self.output_size = tgt
        self.first_linear_dim = embed_dim
        self.ffn_hidden_size = embed_dim
        #self.proj = nn.Linear(embed_dim, tgt)
        self.dropout = nn.Dropout(p=dropout)
        
        self.activation = get_activation_function('ReLU')#['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU']
        
        self.ffn_num_layers=1

        if self.ffn_num_layers == 1:
            ffn = [
                self.dropout,
                nn.Linear(self.first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                self.dropout,
                nn.Linear(self.first_linear_dim, self.ffn_hidden_size)
            ]
            for _ in range(self.ffn_num_layers - 2):
                ffn.extend([
                    self.activation,
                    self.dropout,
                    nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
                ])
            ffn.extend([
                self.activation,
                self.dropout,
                nn.Linear(self.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.proj = nn.Sequential(*ffn)

    def forward(self, x):
        return self.dropout(self.proj(x))

