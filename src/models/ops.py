import numpy as np
import torch
from tclip.CLIP import CLIPModel as TClip
from transformers import CLIPModel, CLIPTokenizer
import clip
import ruclip
from torchvision import transforms

class ClipObject():
    def __init__(self, text_model, image_model, italian=False, device='cuda:0'):
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
    def __init__(self, name="ViT-B/32", device="cuda:0") -> None:
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


class HClipObject():
    def __init__(self, name="ViT-B/32", device="cuda:0") -> None:
        ckpt_mapping = {"ViT-B/16":"openai/clip-vit-base-patch16", 
                    "ViT-B/32":"openai/clip-vit-base-patch32",
                    "ViT-L/14":"openai/clip-vit-large-patch14",
                    "ViT-L/16":"openai/clip-vit-large-patch16"}
        self.clip_model =  CLIPModel.from_pretrained(ckpt_mapping[name]).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(ckpt_mapping[name])
        self.device = device

        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                            std=(0.229, 0.224, 0.225)) 

        self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
        ])
        self.logit_scale = self.clip_model.logit_scale.exp().float()

    def encode_image(self, imgs):
        return self.clip_model.get_image_features(pixel_values = imgs).float().to(self.device)

    def encode_text(self, txts):
        text_inputs = self.tokenizer(txts, padding=True, return_tensors="pt")
        # text_inputs = clip.tokenize(txts).to(self.device)
        text_embeddings =  self.clip_model.get_text_features(input_ids = text_inputs['input_ids'].to(self.device),  attention_mask = text_inputs['attention_mask'].to(self.device)).float()

        return text_embeddings

class RuClipObject():
    def __init__(self, name='ruclip-vit-base-patch32-384', device="cuda:0") -> None:
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

class JapaneseClipObject():
    def __init__(self, model, preprocess, jaclip=None, device="cuda:0") -> None:
        self.model = model
        self.device = device
        self.ja_clip = jaclip
        self.logit_scale = np.exp(0.07)
        self.preprocess = preprocess
        self.tokenizer = self.ja_clip.load_tokenizer()

    def encode_image(self, imgs):
        i_emb = self.model.get_image_features(pixel_values=imgs)
        return i_emb.type(torch.FloatTensor).to(self.device)
    
    def encode_text(self, txts):
        encodings = self.ja_clip.tokenize(
          texts=txts,
          max_seq_len=77,
          device=self.device,
          tokenizer=self.tokenizer, # this is optional. if you don't pass, load tokenizer each time
        )
        txt_emb = self.model.get_text_features(**encodings)
        return txt_emb.to(self.device)

class KoreanClipObject():
    def __init__(self, model, processor, device="cuda:0") -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.logit_scale = np.exp(0.07)
        self.preprocess = None
    
    def encode_image(self, imgs):
        images_c = np.transpose(imgs.cpu().numpy(), (0, 2, 3, 1))
        img_emb = self.model.get_image_features(pixel_values=images_c)
        return torch.from_numpy(np.asarray(img_emb)).type(torch.FloatTensor).to(self.device)

    def encode_text(self, txts):
        inputs = self.processor(
            text=txts,
            return_tensors="jax", # could also be "pt" 
            padding=True
        )
        txt_emb = self.model.get_text_features(inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs['token_type_ids'])
        return torch.from_numpy(np.asarray(txt_emb)).to(self.device)




def load_models(lang, device='cuda:0', large_model=False, model_dir='../results/clips'):
    if large_model:
        # load models
        if lang == 'en' or '2' in lang:
            model = EnglishClipObject(device=device)
        elif lang == 'ru':
            import ruclip
            model = RuClipObject(name='ruclip-vit-base-patch32-224', device=device)
        elif lang == 'ja':
            import japanese_clip as ja_clip
            m, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
            model = JapaneseClipObject(m, preprocess, jaclip = ja_clip ,device=device)
        elif lang == 'ko':
            from models.koclip.koclip import load_koclip
            ko_model, processor = load_koclip("koclip-large")
            model = KoreanClipObject(ko_model, processor, device=device)
        elif lang in ['de', 'es', 'fr']:
            name = 'ViT-B/16' if lang == 'fr' else 'ViT-B/32'
            model = HClipObject(device=device, name=name)
            finetune_ckpt = f'{model_dir}/best_{lang}.pt'
            model.clip_model.load_state_dict(torch.load(finetune_ckpt, map_location=device))
        else: # italian --- reverse
            model = EnglishClipObject(name=f'{model_dir}/best_{lang}.pt', device=device)
        return model, model.logit_scale, model.preprocess
    else:
         # load models
        model_name = None
        model = TClip(lang, model_name, pretrained=False, temperature=0.07, device=device)
        model_path = f'{model_dir}/{clip_data}/best_{lang}.pt'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logit_scale = np.exp(model.temperature)
        return model, logit_scale, None
