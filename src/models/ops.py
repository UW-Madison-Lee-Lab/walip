import numpy as np
import torch, os
import clip
from funcy import chunks
from tqdm import tqdm
import configs
from utils.loader import load_image_data
from utils.helper import save_images
from models.prompt_templates import prompts, generate_texts

from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.append("../coco-clip")
from CLIP import CLIPModel


def get_clip_based_image_embedding(img_data_name, model_params):
    is_eng_clip, _, model = model_params
    image_feature_path = configs.paths['img_dir'] + f'image_feature_{img_data_name}_{configs.flags["using_filtered_images"]}_en{int(is_eng_clip)}_k{configs.num_images}.npy'

    print(" Load the image embedding: " + image_feature_path)
    if os.path.isfile(image_feature_path) and configs.flags["reuse_image_embedding"]:
        image_features = np.load(image_feature_path, allow_pickle=True)
        image_features = torch.Tensor(image_features).cuda()
    else:
        assert model is not None
        image_path = configs.paths['img_dir'] + f'image_{img_data_name}_{configs.flags["using_filtered_images"]}_k{configs.num_images}.npy'
        if os.path.isfile(image_path) and configs.flags["reuse_image_data"]: 
            images = np.load(image_path, allow_pickle=True)
        else:
            images = load_image_data(img_data_name)
            np.save(image_path, images)
        images = torch.Tensor(images)
        save_images(images, f'../results/base_images_{configs.flags["using_filtered_images"]}.png', nrows=10)
        with torch.no_grad():
            image_features = model.image_encoder(images.cuda()).float()
            image_features = model.image_projection(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            np.save(image_feature_path, image_features.cpu().numpy())
    return image_features

def get_batch_clip_based_text_features(text_params, model_params):
    template_size, texts = text_params
    is_eng_clip, tokenizer, model = model_params
    encoded_query = tokenizer(texts, padding=True, truncation=True,max_length=200)
    try:
        batch = {key: torch.tensor(values).cuda() for key, values in encoded_query.items()}
    except:
        from IPython import embed; embed()
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_features = model.text_projection(text_features).float()
    # ensemble
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    bs = text_features.shape[0] // template_size
    text_features = text_features.view(bs, template_size, text_features.shape[-1])
    text_features = text_features.mean(dim=1)
    return text_features

def get_clip_based_text_embedding(txt_data_name, model_params, vocab, lang, mode):
    emb_path = configs.paths['emb_dir'] + 'cliptext_{}_{}_{}.npy'.format(txt_data_name, lang, mode)
    print(" Load the image embedding: " + emb_path)
    if os.path.isfile(emb_path) and configs.flags["reuse_text_embedding"]:
        print('load ', emb_path)
        lang_embs = np.load(emb_path, allow_pickle=True)
    else:
        texts = generate_texts(prompts[lang], vocab, k=configs.num_prompts)
        lang_embs = []
        if configs.num_prompts == -1:
            K = len(prompts[lang])
        else:
            K = configs.num_prompts
        bz = 1024 * K
        for batch_texts in tqdm(chunks(bz, texts)):
            text_params = [K, batch_texts]
            em = get_batch_clip_based_text_features(text_params, model_params)
            lang_embs.append(em.cpu().numpy())
        lang_embs = np.concatenate(lang_embs, axis=0)
        np.save(emb_path, lang_embs) 
    return lang_embs

def get_fingerprint_embedding(image_features, text_features, logit_scale):
    # logits_per_image = logit_scale * image_features @ text_features.t()
    txt_logits = text_features @ image_features.t()
    if configs.num_images > 1:
        K, D = configs.num_images, image_features.shape[0] // configs.num_images
        txt_logits = txt_logits.view(-1, D, K)
        txt_logits = txt_logits.mean(dim=-1)
    probs = (logit_scale * txt_logits).softmax(dim=-1).cpu().detach().numpy()
    return probs


def load_models(lang):
    # load models
    model = CLIPModel(lang).cuda()
    model_path = f'../coco-clip/best_{lang}.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # logit_scale = clip_model.logit_scale.exp().float()
    logit_scale = 1.0
    if lang in ['es', 'en']:
        tokenizer = BertTokenizer.from_pretrained(configs.model_name[lang], do_lower_case=True)
    elif lang == 'it':
        tokenizer = AutoTokenizer.from_pretrained(configs.model_name[lang], do_lower_case=True)
    return lang=='en', model, tokenizer, logit_scale


def load_embedding(emb_type, txt_data_name, img_data_name, lang, vocab=None, mode='test'):
    if emb_type == 'fp':
        fname = f'{emb_type}_{img_data_name}_{configs.flags["using_filtered_images"]}_{txt_data_name}_{lang}_{mode}.npy'
    else:
        fname = f'{emb_type}_{txt_data_name}_{lang}_{mode}.npy'
    print(" Load embedding: " + fname)
    emb_path = configs.paths['emb_dir'] + fname
    if os.path.isfile(emb_path) and configs.flags["reuse_fp_embedding"]:
        print('load ', emb_path)
        emb = np.load(emb_path, allow_pickle=True)
        return emb
    elif emb_type == 'fasttext':
        print("Miss the fast-text embedding")
        emb = np.load(emb_path, allow_pickle=True)
        return emb

    is_eng_clip, model, tokenizer, logit_scale = load_models(lang)
    model_params = (is_eng_clip, tokenizer, model)
    if emb_type == 'fp':
        # load image
        image_features = get_clip_based_image_embedding(img_data_name, model_params)
        # load text 
        text_features = get_clip_based_text_embedding(txt_data_name, model_params, vocab, lang, mode)
        text_features = torch.from_numpy(text_features).cuda()
        # get emb
        emb = get_fingerprint_embedding(image_features, text_features, logit_scale)
    elif emb_type == 'cliptext':
        emb = get_clip_based_text_embedding(txt_data_name, model_params, vocab, lang, mode)

    np.save(emb_path, emb) 
    return emb


