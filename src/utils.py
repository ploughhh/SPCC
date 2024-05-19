import os

import numpy as np
import torch
from numba import njit


def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset)


def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'sc_spa_model'
    return name + '_' + args.model

def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pretrained_model/{name}.pt')

def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pretrained_model/{name}.pt')
    return model

