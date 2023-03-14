import os
import logging
from config import config, BaseConfig
from typing import Any, List, Optional, Union

import torch
import gradio as gr
from speechbrain.pretrained import EncoderClassifier
from TweakedEncoderClassifier import *

''' Gradio Input/Output Configurations '''
inputs: Union[str, gr.Audio] = gr.Audio(source='upload', type='filepath')
outputs: gr.HighlightedText = gr.HighlightedText()

''' CPU/GPU Configurations '''
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = 'gpu'
else:
    DEVICE = 1
    ACCELERATOR = 'cpu'

MAP_LOCATION: str = torch.device('cuda:{}'.format(DEVICE[0]) if ACCELERATOR == 'gpu' else 'cpu')


''' Helper functions '''
def initialize_lid_model(cfg: BaseConfig) -> EncoderClassifier:

    # lid_model = EncoderClassifier.from_hparams(source=cfg.model_source, savedir=cfg.model_dir)
    lid_model = TweakedEncoderClassifier.from_hparams(source=cfg.model_source, savedir=cfg.model_source)

    return lid_model

''' Initialize models '''
lid_model = initialize_lid_model(config)

''' Main prediction function '''
def predict(input, wav=False) -> str:
    if wav is False:
        signal = lid_model.load_audio(input)
    else:
        signal = input
    print(signal.shape)
    prediction =  lid_model(signal)
    # language = prediction[3]

    pscore, index, language = lid_model.postproc(prediction)

    return [(language, 'Language')]

def trace(model, output_path):
    input = torch.rand([20800])
    traced_model = torch.jit.trace(model, input)
    return traced_model.save(output_path)