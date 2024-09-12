import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnFunctional
from torch.autograd import Variable
import pickle

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = torch.nn.functional.gelu  #utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    
    
def build_model(tgt,embed_dim=1024, dropout=0.05, out_both=False,min_max_key=dict([('pressure', [-4.0, 8.0]), ('temperature', [50.0, 470.0])])):
    c = copy.deepcopy
    
    model = Encoder3D(XRDModel(input_dim=1701,inner_dim = 512,inner_dim2 = 1024,last_dropout=dropout), RegressionHead(embed_dim+384,512, tgt, last_dropout=dropout), out_both,min_max_key)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Encoder3D(nn.Module):
    def __init__(self, encoder,  generator, out_both,min_max_key):
        super(Encoder3D, self).__init__()
        self.encoder = encoder
        self.env_embed = EnvModel(hidden_dim=128, bins=32,min_max_key=min_max_key)   #shape = 16,384  384=128+128+128
        
        self.generator = generator
        self.out_both = out_both

    def forward_once(self, xrd_input,pressure, temperature):
        self.dist = dist
        xrd_encoder_ed = self.encoder(self.src_pe(self.src_embed(src), pos), dist.unsqueeze(1), src_mask)[:, 0, :]
        env_embed_ed = self.env_embed(pressure, temperature)
        all_encoder = torch.cat([xrd_encoder_ed, env_embed_ed], dim=-1)
        return all_encoder

    def forward(self, xrd_input, pressure, temperature):
        if self.out_both:
            h = self.forward_once(xrd_input, pressure, temperature)
            return self.generator(h), h
        return self.generator(self.forward_once(xrd_input, pressure, temperature))

class XRDModel(nn.Module):

    def __init__(
        self,
        input_dim=1701,
        inner_dim = 512,
        inner_dim2 = 1024,
        last_dropout=0.05,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)  # 512+128*5  ，  128*2
        self.dense2 = nn.Linear(inner_dim, inner_dim2)  # 512+128*5  ，  128*2
        self.activation_fn =  torch.nn.functional.relu  #  #utils.get_activation_fn(activation_fn)  #relu
        self.dropout = nn.Dropout(p=last_dropout)  #   0.0

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        #x = self.dropout(x)
        return x

        
class EnvModel(nn.Module):
    def __init__(self, hidden_dim=128, bins=32, min_max_key=None):
        super().__init__()
        self.project = NonLinearHead(2, hidden_dim, 'relu')
        self.bins = bins
        self.pressure_embed = nn.Embedding(bins, hidden_dim)
        self.temperature_embed = nn.Embedding(bins, hidden_dim)
        self.min_max_key = min_max_key
    def forward(self, pressure, temperature):
        pressure = pressure.type_as(self.project.linear1.weight)
        temperature = temperature.type_as(self.project.linear1.weight)
        #print(self.min_max_key,type(self.min_max_key))
        pressure = torch.clamp(pressure, self.min_max_key['pressure'][0], self.min_max_key['pressure'][1])
        temperature = torch.clamp(temperature, self.min_max_key['temperature'][0], self.min_max_key['temperature'][1])
        pressure = (pressure - self.min_max_key['pressure'][0]) / (self.min_max_key['pressure'][1] - self.min_max_key['pressure'][0])
        temperature = (temperature - self.min_max_key['temperature'][0]) / (self.min_max_key['temperature'][1] - self.min_max_key['temperature'][0])
        # shapes of pressure and temperature both are [batch_size, ]
        env_project = torch.cat((pressure[:, None], temperature[:, None]), dim=-1)
        env_project = self.project(env_project)  # shape of env_project is [batch_size, env_dim]

        pressure_bin = torch.floor(pressure * self.bins).to(torch.long)
        temperature_bin = torch.floor(temperature * self.bins).to(torch.long)
        pressure_embed = self.pressure_embed(pressure_bin)  # shape of pressure_embed is [batch_size, env_dim]
        temperature_embed = self.temperature_embed(temperature_bin)  # shape of temperature_embed is [batch_size, env_dim]
        env_embed = torch.cat([pressure_embed, temperature_embed], dim=-1)
        #print(pressure_embed.shape,temperature_embed.shape,env_project.shape)

        env_repr = torch.cat([env_project, env_embed], dim=-1)

        return env_repr

class RegressionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        inner_dim,
        tgt,
        last_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)  # 512+128*5  ，  128*2
        self.activation_fn =  torch.nn.functional.relu  #  #utils.get_activation_fn(activation_fn)  #relu
        self.dropout = nn.Dropout(p=last_dropout)  #   0.0
        self.out_proj = nn.Linear(inner_dim, tgt)  #  128*2，num_classes

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
