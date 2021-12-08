import numpy as np
import torch, os
import clip
from funcy import chunks
from tqdm import tqdm
import configs
from utils.loader import load_image_data
from utils.helper import save_images
from models.prompt_templates import prompts, generate_texts
from models.clip_italian import get_italian_models


def get_clip_based_image_embedding(img_data_name, model_params):
    is_eng_clip, model = model_params
    image_feature_path = configs.image_feature_prefix + '{}_en{}_k{}.npy'.format(img_data_name, int(is_eng_clip), configs.num_images)
    if os.path.isfile(image_feature_path) and configs.flags["reuse_image_embedding"]:
        image_features = np.load(image_feature_path, allow_pickle=True)
        image_features = torch.Tensor(image_features).cuda()
    else:
        assert model is not None
        image_path = configs.image_prefix + '{}_k{}.npy'.format(img_data_name, configs.num_images)
        if os.path.isfile(image_path) and configs.flags["reuse_image_data"]: 
            images = np.load(image_path, allow_pickle=True)
        else:
            images = load_image_data(img_data_name)
            np.save(image_path, images)
        images = torch.Tensor(images)
        save_images(images, '../results/base_images.png', nrows=10)
        with torch.no_grad():
            if is_eng_clip:
                image_features = model.encode_image(images.cuda()).float()
            else:
                image_features = model(images)
                image_features = torch.from_numpy(image_features).cuda()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            np.save(image_feature_path, image_features.cpu().numpy())
    return image_features

def get_batch_clip_based_text_features(text_params, model_params):
    template_size, texts = text_params
    is_eng_clip, model = model_params
    if is_eng_clip:
        text_tokens = clip.tokenize(texts).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
    else:
        with torch.no_grad():
            text_features = model(texts)
            text_features = torch.from_numpy(text_features)
        text_features = text_features.cuda()
    # ensemble
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    bs = text_features.shape[0] // template_size
    text_features = text_features.view(bs, template_size, text_features.shape[-1])
    text_features = text_features.mean(dim=1)
    return text_features

def get_clip_based_text_embedding(txt_data_name, model_params, vocab, lang, mode):
    emb_path = configs.text_prefix + 'cliptext_{}_{}_{}.npy'.format(txt_data_name, lang, mode)
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

def get_fingerprint_embedding(image_features, text_features, logit_scale=1.0):
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
    if lang == 'en':
        is_eng_clip = True
        clip_model, preprocess = clip.load("ViT-B/32")
        clip_model.cuda().eval()
        logit_scale = clip_model.logit_scale.exp().float()
        image_model = text_model = clip_model
    else: # italian
        is_eng_clip = False
        # text_model = multilingual_clip.load_model('M-BERT-Base-ViT-B')
        # text_model.eval()
        image_model, text_model = get_italian_models()
        logit_scale = 1.0
    return is_eng_clip, image_model, text_model, logit_scale

def load_embedding(emb_type, txt_data_name, img_data_name, lang, vocab=None, mode='test'):
    if emb_type == 'fp':
        emb_path = configs.emb_prefix + '{}_{}_{}_{}_{}.npy'.format(emb_type, img_data_name, txt_data_name, lang, mode)
    else:
        emb_path = configs.emb_prefix + '{}_{}_{}_{}.npy'.format(emb_type, txt_data_name, lang, mode)
    if os.path.isfile(emb_path) and configs.flags["reuse_fp_embedding"]:
        print('load ', emb_path)
        emb = np.load(emb_path, allow_pickle=True)
        return emb
    elif emb_type == 'fasttest':
        print("Miss the fast-text embedding")
        return -1

    is_eng_clip, image_model, text_model, logit_scale = load_models(lang)
    if emb_type == 'fp':
        img_model_params = (is_eng_clip, image_model)
        txt_model_params = (is_eng_clip, text_model)
        # load image
        image_features = get_clip_based_image_embedding(img_data_name, img_model_params)
        # load text 
        text_features = get_clip_based_text_embedding(txt_data_name, txt_model_params, vocab, lang, mode)
        text_features = torch.from_numpy(text_features).cuda()
        # get emb
        emb = get_fingerprint_embedding(image_features, text_features, logit_scale)
    elif emb_type == 'cliptext':
        txt_model_params = (is_eng_clip, text_model)
        emb = get_clip_based_text_embedding(txt_data_name, txt_model_params, vocab, lang, mode)

    np.save(emb_path, emb) 
    return emb


