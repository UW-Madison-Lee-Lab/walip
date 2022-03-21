# from this import d
import torch
from tclip.CLIP import CLIPModel
import numpy as np
import clip
import ruclip
from koclip import load_koclip
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Any, Union, List
from tclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn.utils.rnn import pad_sequence

def italian_tokenize(_tokenizer, tokenizer, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result



class ClipObject():
    def __init__(self, text_model, image_model, italian=False, device='cuda'):
        self.text_model = text_model
        self.image_model = image_model
        self.italian = italian
        self.device = device
    
    def encode_text(self, txt):
        if self.italian:
            text_features = self.text_model(txt)
            text_features = torch.from_numpy(text_features).to(self.device)
        else:
            text_features = self.text_model(txt).to(self.device)
        return text_features

    def encode_image(self, imgs):
        if self.italian:
            image_features = self.image_model(imgs.cpu())
            image_features = torch.from_numpy(image_features).to(self.device)
        else:
            image_features = self.image_model.encode_image(imgs).type(torch.FloatTensor).to(self.device)
        return image_features

class EnglishClipObject():
    def __init__(self, name="ViT-B/32", device="cuda") -> None:
        self.clip_model, self.preprocess = clip.load(name)
        self.clip_model = self.clip_model.to(device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().float()
        self.device = device
    
    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs).type(torch.FloatTensor).to(self.device)

    def encode_text(self, txts):
        text_tokens = clip.tokenize(txts).to(self.device)
        text_embeddings = self.clip_model.encode_text(text_tokens).type(torch.FloatTensor).to(self.device)
        return text_embeddings

class RuClipObject():
    def __init__(self, name='ruclip-vit-base-patch32-384', device="cuda") -> None:
        self.clip_model, self.clip_processor = ruclip.load(name, device=device)
        self.clip_model = self.clip_model.to(device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().float()
        self.preprocess = self.clip_processor.image_transform
        self.device = device
    
    def encode_image(self, imgs):
        image_latents = self.clip_model.encode_image(imgs.to(self.device))
        image_embeddings = image_latents / image_latents.norm(dim=-1, keepdim=True)
        return image_embeddings

    def encode_text(self, txts):
        inputs = self.clip_processor(text=txts, return_tensors='pt', padding=True)
        text_embeddings = self.clip_model.encode_text(inputs['input_ids'].to(self.device))
        # .type(torch.FloatTensor)
        return text_embeddings

class ItalianClipObject():
    def __init__(self, name="ViT-B/32", device="cuda") -> None:
        self.clip_model, self.preprocess = clip.load(name)
        self.clip_model = self.clip_model.to(device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().float()
        self.device = device
        self._tokenizer = _Tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    
    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs).type(torch.FloatTensor).to(self.device)

    def encode_text(self, txts):
        text_tokens = italian_tokenize(self._tokenizer, self.tokenizer, txts).to(self.device)
        text_embeddings = self.clip_model.encode_text(text_tokens).type(torch.FloatTensor).to(self.device)
        return text_embeddings

class KoreanClipObject():
    def __init__(self, name="koclip-base", device="cuda") -> None:
        self.clip_model, self.processor = load_koclip(name)
        self.device = device
    
    def encode_image(self, imgs):
        imgs = np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1))
        inputs = self.processor(
        text=["hello"], # Can put any text because just looking at image embed
        images=[img for img in imgs], 
        return_tensors="jax",
        padding=True
    )

        outputs = self.clip_model(**inputs)
        return torch.FloatTensor(np.array(outputs['image_embeds'])).to(self.device)

    def encode_text(self, txts):
        inputs = self.processor(
        text=txts,
        images=np.zeros((10,10,3)), # Can put any image because just looking at text
        return_tensors="jax",
        padding=True
        )   

        outputs = self.clip_model(**inputs)
        return torch.FloatTensor(np.array(outputs['text_embeds'])).to(self.device)

        
def load_models(lang, model_name, clip_data='coco', device='cuda', large_model=False):
    if not large_model:
        # load models
        model = CLIPModel(lang, model_name, pretrained=False, temperature=0.07, device=device)
        model = model.to(device)
        model_path = f'../results/clips/{clip_data}/best_{lang}.pt'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logit_scale = np.exp(model.temperature)
        return model, logit_scale, None
    else:
        # load models
        if lang == 'en':
            model = EnglishClipObject(device=device)
            return model, model.logit_scale, model.preprocess
        elif lang == 'en2':
            model = EnglishClipObject(name='RN50x4')
            return model, model.logit_scale, model.preprocess
        elif lang == 'sw':
            from mclip import multilingual_clip
            text_model = multilingual_clip.load_model('Swe-CLIP-2M')
            # text_model = multilingual_clip.load_model('M-BERT-Distil-40')
            img_model, preprocess = clip.load('RN50x4')
            text_model = text_model.eval()
            img_model = img_model.to(device).eval()
            logit_scale = img_model.logit_scale.exp().float()
            clip_model = ClipObject(text_model, img_model, device=device)
        elif lang == 'it': # italian
            model = EnglishClipObject(name='../results/clips/wtm/it.pt')
            # model = EnglishClipObject()
            return model, model.logit_scale, model.preprocess
            preprocess = None
            from models.clip_italian import get_italian_models
            img_model, text_model = get_italian_models()
            clip_model = ClipObject(text_model, img_model, italian=True, device=device)
            logit_scale = 20.0
        elif lang == 'ru':
            model = RuClipObject(name='ruclip-vit-base-patch32-224')
            # model = EnglishClipObject()
            return model, model.logit_scale, model.preprocess
        elif lang == 'ko':
            model = KoreanClipObject(name="koclip-base")
            return model, 1.0, None

        return clip_model, logit_scale, preprocess