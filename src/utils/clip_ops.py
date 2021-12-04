# -*- coding: utf-8 -*-

import os 
import sys
import json
import zipfile
import natsort

import numpy as np
import pandas as pd

from PIL import Image as PilImage
os.environ['TOKENIZERS_PARALLELISM'] = "false"

import transformers
from transformers import AutoTokenizer

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from tqdm.notebook import tqdm

!wget -q -N https://github.com/huggingface/transformers/raw/master/examples/research_projects/jax-projects/hybrid_clip/modeling_hybrid_clip.py
!wget -q -N https://github.com/huggingface/transformers/raw/master/examples/research_projects/jax-projects/hybrid_clip/configuration_hybrid_clip.py

sys.path.append('.')

from modeling_hybrid_clip import FlaxHybridCLIP

# Model selection
MODEL_TYPE = 'clip_italian'

!git clone https://github.com/crux82/mscoco-it/
!wget -N -q --show-progress http://images.cocodataset.org/zips/val2014.zip
!gunzip /content/mscoco-it/mscoco-it/captions_ita_devset_validated.json.gz
!gunzip /content/mscoco-it/mscoco-it/captions_ita_devset_unvalidated.json.gz

img_folder = 'photos/'

if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)


for img in destroy_images:
    os.remove(img)

if MODEL_TYPE == 'mClip':
    from sentence_transformers import SentenceTransformer
    # Here we load the multilingual CLIP model. Note, this model can only encode text.
    # If you need embeddings for images, you must load the 'clip-ViT-B-32' model
    se_language_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
    se_image_model = SentenceTransformer('clip-ViT-B-32')
    language_model = lambda queries: se_language_model.encode(queries, convert_to_tensor=True, show_progress_bar=False).cpu().detach().numpy()
    image_model = lambda images: se_image_model.encode(images, batch_size=1024, convert_to_tensor=True, show_progress_bar=False).cpu().detach().numpy()
elif MODEL_TYPE == 'clip_italian':
    import jax
    from jax import numpy as jnp
    TOKENIZER_NAME = "dbmdz/bert-base-italian-xxl-uncased"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=None, use_fast=True)
    model = FlaxHybridCLIP.from_pretrained("clip-italian/clip-italian")
    def tokenize(texts):
        inputs = tokenizer(texts, max_length=96, padding="max_length", return_tensors="np")
        return inputs['input_ids'], inputs['attention_mask']

    language_model = lambda queries: np.asarray(model.get_text_features(*tokenize(queries)))
    image_model = lambda images: np.asarray(model.get_image_features(images.permute(0, 2, 3, 1).numpy(),))

"""# Utils"""

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def get_image_name(self, idx):
        return self.total_imgs[idx]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PilImage.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

class SimpleTextDataset(torch.utils.data.Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def text_encoder(text):
    # inputs = tokenizer([text], max_length=96, truncation=True, padding="max_length", return_tensors="np")
    # embedding = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])[0]
    embedding = language_model(text)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def precompute_text_features(loader):
    text_features = []

    for i, (texts) in enumerate(tqdm(loader)):
        # inputs = tokenizer(texts, max_length=96, truncation=True, padding="max_length", return_tensors="np")
        # embedding = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
        # embedding /= jnp.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = language_model(texts)
        embedding = embedding / np.linalg.norm(embedding)

        text_features.extend(embedding)

    return np.array(text_features)

def precompute_image_features(loader):
    image_features = []
    for i, (images) in enumerate(tqdm(loader)):
        # images = images.permute(0, 2, 3, 1).numpy()
        # features = model.get_image_features(images,)
        features =image_model(images)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        image_features.extend(features)
    return np.array(image_features)

def find_image(text_query, datatset, image_features, n=1):
    zeroshot_weights = text_encoder(text_query)
    zeroshot_weights = zeroshot_weights / np.linalg.norm(zeroshot_weights)
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append('photos/val2014/' + dataset.get_image_name(idx))
    return file_paths

def show_images(image_list):
    for im_path in image_list:
        display(Image(filename=im_path))

image_size = 224

val_preprocess = transforms.Compose([
    Resize([image_size], interpolation=InterpolationMode.BICUBIC),
    CenterCrop(image_size),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

dataset = CustomDataSet("photos/val2014", transform=val_preprocess)
text_dataset = SimpleTextDataset([elem["caption"] for elem in data])

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    persistent_workers=True,
    drop_last=False)

text_loader = torch.utils.data.DataLoader(
    text_dataset,
    batch_size=1024,
    shuffle=False)

image_features = precompute_image_features(loader)

text_features = precompute_text_features(text_loader)

def compute_mrr(data, dataset, n):
    collect_rr = []

    pbar = tqdm(total=len(data), position=0, leave=True)

    found = np.matmul(text_features, image_features.T)
    for index, distances in enumerate(found):
        pbar.update(1)
        image_path = get_path_coco(data[index]["image_id"])
        collect_rr.append(new_rr(distances, image_path, dataset, n))

    pbar.close()
    return np.average(collect_rr)


def new_rr(distances, target_image, dataset, n):
    image_paths = []
    idxs = distances.argsort()[-n:][::-1]
    for idx in idxs:
        image_paths.append('photos/val2014/' + dataset.get_image_name(idx))

    if target_image in image_paths:
        return 1/(image_paths.index(target_image) + 1)
    else:
        return 0

    return 1/rank

def internal_hits(distances, target_image, dataset, n):
    image_paths = []
    idxs = distances.argsort()[-n:][::-1]
    for idx in idxs:
        image_paths.append('photos/val2014/' + dataset.get_image_name(idx))

    if target_image in image_paths:
        return 1
    else:
        return 0

def compute_hits(data, dataset, n):
    collect_rr = []

    pbar = tqdm(total=len(data), position=0, leave=True)

    found = np.matmul(text_features, image_features.T)
    for index, distances in enumerate(found):
        pbar.update(1)
        image_path = get_path_coco(data[index]["image_id"])
        collect_rr.append(internal_hits(distances, image_path, dataset, n))

    pbar.close()
    return np.average(collect_rr)

print('MRR@1:', compute_mrr(data, dataset, 1))
print('MRR@5:', compute_mrr(data, dataset, 5))
print('MRR@10:', compute_mrr(data, dataset, 10))

compute_hits(data, dataset, 100)

data[215]["caption"]

text = "La vista frontale di un edificio che ha una panchina, e sedie situate di fronte"

image_paths = find_image(text, dataset, image_features, n=3)
show_images(image_paths)



"""# UNSPLASH"""

from sentence_transformers import SentenceTransformer, util
import glob
import torch
import pickle
import zipfile
import os
import zipfile
import json
import os
from tqdm.autonotebook import tqdm

img_folder = 'unsplash/'
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)
    
    photo_filename = 'unsplash-25k-photos.zip'
    if not os.path.exists(photo_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+photo_filename, photo_filename)
        
    #Extract all images
    with zipfile.ZipFile(photo_filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, img_folder)

dataset = CustomDataSet("unsplash", transform=val_preprocess)


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    persistent_workers=True,
    drop_last=False)

image_features = precompute_image_features(loader)

def find_image(text_query, datatset, image_features, n=1):
    zeroshot_weights = text_encoder(text_query)
    zeroshot_weights = zeroshot_weights / np.linalg.norm(zeroshot_weights)
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append('unsplash/' + dataset.get_image_name(idx))
    return file_paths

def show_images(image_list):
    for im_path in image_list:
        display(Image(filename=im_path))

text = "un cane nero"

image_paths = find_image(text, dataset, image_features, n=3)
show_images(image_paths)








