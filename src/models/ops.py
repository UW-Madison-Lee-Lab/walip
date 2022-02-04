# from this import d
import torch

from tclip.CLIP import CLIPModel
import numpy as np
import clip
import torch.nn.functional as F

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
            text_features = self.text_model(txt)
        return F.normalize(text_features, dim=-1)

    def encode_image(self, imgs):
        if self.italian:
            image_features = self.image_model(imgs.cpu())
            image_features = torch.from_numpy(image_features).to(self.device)
        else:
            image_features = self.image_model.encode_image(imgs).type(torch.cuda.FloatTensor)
        return F.normalize(image_features, dim=-1)

class EnglishClipObject():
    def __init__(self, name="ViT-B/32", device="cuda") -> None:
        self.clip_model, self.preprocess = clip.load(name)
        self.clip_model = self.clip_model.to(device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().float()
        self.device = device
    
    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs).type(torch.cuda.FloatTensor)

    def encode_text(self, txts):
        text_tokens = clip.tokenize(txts).to(self.device)
        text_embeddings = self.clip_model.encode_text(text_tokens).type(torch.cuda.FloatTensor)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        return text_embeddings


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
            model = EnglishClipObject()
            return model, model.logit_scale, model.preprocess
        elif lang == 'sw':
            from mclip import multilingual_clip
            model = multilingual_clip.load_model('Swe-CLIP-2M')
            text_model = multilingual_clip.load_model('M-BERT-Distil-40')
            img_model, preprocess = clip.load('RN50x4')
            text_model = text_model.eval()
            img_model = img_model.to(device).eval()
            logit_scale = img_model.logit_scale.exp().float()
            clip_model = ClipObject(text_model, img_model, device=device)
        else: # italian
            preprocess = None
            from models.clip_italian import get_italian_models
            img_model, text_model = get_italian_models()
            clip_model = ClipObject(text_model, img_model, italian=True, device=device)
            logit_scale = 20.0

        return clip_model, logit_scale, preprocess