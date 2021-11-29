import numpy as np
import torch, os
import clip
from transformers import AutoTokenizer
from mclip import multilingual_clip
from modeling_hybrid_clip import FlaxHybridCLIP
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def get_clip_image_features(clip_name, image_name, num_images=1, model=None, preprocess=None):
    if clip_name == "ViT-B/32":
        clip_name = 'vit32'
    fname = '{}_{}_{}.npy'.format(image_name, num_images, clip_name)
    image_feature_path = '../data/image_feature_' + fname
    image_path = '../data/image_' + fname
    if os.path.isfile(image_feature_path):
        image_features = np.load(image_feature_path, allow_pickle=True)
        image_features = torch.Tensor(image_features).cuda()
    else:
        assert model is not None
        if os.path.isfile(image_path):
            images = np.load(image_path, allow_pickle=True)
        else:
            assert preprocess is not None
            if image_name == 'cifar100':
                from torchvision.datasets import CIFAR100
                image_dataset = CIFAR100('../data', transform=preprocess, download=True)
            else:
                import h5py as h5
                from PIL import Image
                hdf5_path = '../../repgan/data/tiny/tiny_valid.h5'
                print('Loading %s into memory...' % hdf5_path)
                with h5.File(hdf5_path, 'r') as f:
                    imgs = np.asarray(f.get('imgs'))
                    labels = np.asarray(f.get('labels'))
            images = []
            if image_name == 'cifar100':
                num_classes = 100
                for c in range(num_classes):
                    indices = np.argwhere(np.asarray(image_dataset.targets) == c)
                    for i in range(num_images):
                        images.append(image_dataset[indices[i][0]])
            else:
                num_classes = 200
                for c in range(num_classes):
                    indices = np.argwhere(labels == c)
                    for i in range(num_images):
                        image = imgs[indices[i][0]]
                        image = Image.fromarray(image, 'RGB')
                        image = preprocess(image)
                        images.append(image)

            images = np.stack(images, axis=0)
            np.save(image_path, images)
        images = torch.Tensor(images).cuda()
        with torch.no_grad():
            image_features = model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            np.save(image_feature_path, image_features.cpu().numpy())
    return image_features

def get_clip_text_features(texts, model, is_clip=True):
    if is_clip:
        text_tokens = clip.tokenize(texts).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
    else:
        with torch.no_grad():
            text_features = model(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def cal_probs_from_features(image_features, text_features, logit_scale):
    # logits_per_image = logit_scale * image_features @ text_features.t()
    txt_logits = logit_scale * (text_features @ image_features.t())
    probs = txt_logits.softmax(dim=-1).cpu().detach().numpy()
    return probs


def get_fingerprints(texts, clip_name, is_clip=True, image_name='tiny', num_images=1):
    # clip & images
    clip_model, preprocess = clip.load(clip_name)
    clip_model.cuda().eval()
    image_features = get_clip_image_features(clip_name, image_name, num_images, clip_model, preprocess)
    if is_clip:
        text_features = get_clip_text_features(texts, clip_model, is_clip=True)
    else:
        # text_model = multilingual_clip.load_model('M-BERT-Distil-40')
        text_model = multilingual_clip.load_model('M-BERT-Base-ViT-B')
        text_model.eval()
        text_features = get_clip_text_features(texts, text_model, is_clip=False)

    # CLIP Temperature scaler
    logit_scale = clip_model.logit_scale.exp().float()
    probs = cal_probs_from_features(image_features, text_features, logit_scale)
    return probs


def text_encoder(language_model, text):
    # inputs = tokenizer([text], max_length=96, truncation=True, padding="max_length", return_tensors="np")
    # embedding = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])[0]
    embedding = language_model(text)
    # embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding

def precompute_text_features(language_model, texts):
    # inputs = tokenizer(texts, max_length=96, truncation=True, padding="max_length", return_tensors="np")
    # embedding = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
    # embedding /= jnp.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = language_model(texts)
    embedding = embedding / np.linalg.norm(embedding)
    # embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return np.asarray(embedding)

def precompute_image_features(image_model, images):
    embedding =image_model(images)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    # embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return np.asarray(embedding)

def get_models(MODEL_TYPE):
    if MODEL_TYPE == 'mClip':
        from sentence_transformers import SentenceTransformer
        # Here we load the multilingual CLIP model. Note, this model can only encode text.
        # If you need embeddings for images, you must load the 'clip-ViT-B-32' model
        se_language_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        se_image_model = SentenceTransformer('clip-ViT-B-32')
        language_model = lambda queries: se_language_model.encode(queries, convert_to_tensor=True, show_progress_bar=False).cpu().detach().numpy()
        image_model = lambda images: se_image_model.encode(images, batch_size=1024, convert_to_tensor=True, show_progress_bar=False).cpu().detach().numpy()
    elif MODEL_TYPE == 'clip_italian':
        TOKENIZER_NAME = "dbmdz/bert-base-italian-xxl-uncased"
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=None, use_fast=True)
        model = FlaxHybridCLIP.from_pretrained("clip-italian/clip-italian")
        def tokenize(texts):
            inputs = tokenizer(texts, max_length=96, padding="max_length", return_tensors="np")
            return inputs['input_ids'], inputs['attention_mask']

        language_model = lambda queries: model.get_text_features(*tokenize(queries))
        image_model = lambda images: model.get_image_features(images.permute(0, 2, 3, 1),)
    
    return language_model, image_model