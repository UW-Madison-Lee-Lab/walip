import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer
from tclip.CLIP import CLIPModel
import numpy as np

def get_tokenizer(lang, model_name):
    if lang in ['es', 'en']:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    elif lang == 'it':
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    return tokenizer


def load_models(lang, model_name, clip_data='coco', device='cuda'):
    # load models
    model = CLIPModel(lang, model_name, pretrained=False, temperature=0.07)
    model = model.to(device)
    model_path = f'../results/clips/{clip_data}/best_{lang}.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logit_scale = np.exp(model.temperature)
    return model, logit_scale
