#!/usr/bin/env python
# coding: utf-8

import os
import torch
from AssistiveOptimization import AssistiveTexturization, run_texturization
import warnings
warnings.filterwarnings("ignore")


settings = {
    'out_path': 'outputs',
    'n_views': 5,
    'n_iter': 40,
    'lr': 5e-2,
    'tv_loss':1e-6,
    'image_size': 512,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'meshes': { 
        'jeep':{
           'init_color': [128.,128.,128.], # Initial RGB color
            'inter_cam': [2.0, 10, 15, 0, 260],
            'target': 609, # ImageNet Label
        },
        'lexus':{
           'init_color': [128.,128.,128.], # Initial RGB color
            'inter_cam': [2.0, -80, -20, 0, 0],
            'target': 717, # ImageNet Label
        }
    },
    # Test models
    'model_names': ['resnet','squeezenet']
}

# Run Texturization
run_texturization(settings)





