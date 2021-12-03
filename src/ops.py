import numpy as np
import torch, os
import clip
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from funcy import chunks
from tqdm import tqdm

from mclip import multilingual_clip
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# from modeling_hybrid_clip import FlaxHybridCLIP
means = {
	'cifar10': (0.4914, 0.4822, 0.4465),
	'cifar100': (0.5071, 0.4867, 0.4408),
	'imagenet': (0.48145466, 0.4578275, 0.40821073),
	'tiny': (0.48145466, 0.4578275, 0.40821073)

}

stds = {
	'cifar10': (0.2023, 0.1994, 0.2010),
	'cifar100': (0.2675, 0.2565, 0.2761),
	'imagenet': (0.26862954, 0.26130258, 0.27577711),
	'tiny': (0.26862954, 0.26130258, 0.27577711)
}


def load_image_data(image_name, num_images=1):
    preprocess = transforms.Compose([
        Resize([224], interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(means[image_name], stds[image_name]),
    ])
    if image_name == 'cifar100':
        from torchvision.datasets import CIFAR100
        image_dataset = CIFAR100('../data', transform=preprocess, download=True, train=False)
    elif image_name == 'cifar10':
        from torchvision.datasets import CIFAR10
        image_dataset = CIFAR10('../data', transform=preprocess, download=True, train=False)
    elif image_name == 'tiny':
        import h5py as h5
        from PIL import Image
        hdf5_path = '../../repgan/data/tiny/tiny_valid.h5'
        print('Loading %s into memory...' % hdf5_path)
        with h5.File(hdf5_path, 'r') as f:
            imgs = np.asarray(f.get('imgs'))
            labels = np.asarray(f.get('labels'))
    else:
        from torchvision.datasets import ImageFolder
        data_dir = '/mnt/nfs/work1/mccallum/dthai/tuan_data/imagenet/val'
        image_dataset = ImageFolder(data_dir, preprocess)

    fpath = '../dicts/npy/image_{}_{}_{}_index.npy'.format(image_name, 'en', 'it')
    if os.path.isfile(fpath):
        dct = np.load(fpath, allow_pickle=True).item()
        indices = list(dct.values())
        images = [image_dataset[idx][0] for idx in indices]
    else:
        images = []
        # pick randomly
        if image_name == 'cifar100':
            with open('../dicts/texts/cifar100_index.txt') as f:
                lines = f.readlines()
            d = {}
            for l in lines:
                k, v = l.strip().split(' ')
                k, v = int(k), int(v)
                if k in d:
                    d[k].append(v)
                else:
                    d[k] = [v]
            num_classes = 100
            for c in range(num_classes):
                # indices = np.argwhere(np.asarray(image_dataset.targets) == c)
                # for i in range(num_images):
                    # images.append(image_dataset[indices[i][0]][0])
                for i in range(num_images):
                    idx = d[c][i]
                    images.append(image_dataset[idx][0])
        elif image_name == 'cifar10':
            num_classes = 10
            for c in range(num_classes):
                indices = np.argwhere(np.asarray(image_dataset.targets) == c)
                for i in range(num_images):
                    images.append(image_dataset[indices[i][0]][0])
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
    return images

def get_clip_image_features(clip_name, image_name, num_images=1, model=None):
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
            images = load_image_data(image_name, num_images)
            np.save(image_path, images)
        images = torch.Tensor(images).cuda()
        with torch.no_grad():
            image_features = model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            np.save(image_feature_path, image_features.cpu().numpy())
    return image_features

def get_clip_text_features(texts, model, is_clip=True, K=1):
    if is_clip:
        text_tokens = clip.tokenize(texts).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
    else:
        with torch.no_grad():
            text_features = model(texts)
        text_features = text_features.cuda()
    # ensemble
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    B = text_features.shape[0] // K
    text_features = text_features.view(B, K, text_features.shape[-1])
    text_features = text_features.mean(dim=1)
    return text_features

def cal_probs_from_features(image_features, text_features, logit_scale):
    # logits_per_image = logit_scale * image_features @ text_features.t()
    # logit_scale = 1
    txt_logits = logit_scale * (text_features.cuda() @ image_features.t())
    # probs = txt_logits.cpu().detach().numpy()
    probs = txt_logits.softmax(dim=-1).cpu().detach().numpy()
    return probs


clip_names = {'en': "ViT-B/32",
        # 'it': 'RN50x4'
        'it': 'ViT-B/32'
    }

def get_fingerprints(texts, lang, image_name='tiny', num_images=1, K=-1, text_embedding=False):
    clip_model, preprocess = clip.load(clip_names[lang])
    clip_model.cuda().eval()
    if lang == 'en':
        text_features = get_clip_text_features(texts, clip_model, True, K)
    else:
        # text_model = multilingual_clip.load_model('M-BERT-Distil-40')
        text_model = multilingual_clip.load_model('M-BERT-Base-ViT-B')
        text_model.eval()
        text_features = get_clip_text_features(texts, text_model, False, K)

    # CLIP Temperature scaler
    if text_embedding:
        return text_features
    else:
        image_features = get_clip_image_features(clip_names[lang], image_name, num_images, clip_model)
        logit_scale = clip_model.logit_scale.exp().float()
        probs = cal_probs_from_features(image_features, text_features, logit_scale)
        return probs


def get_models(MODEL_TYPE):
    if MODEL_TYPE == 'mClip':
        from sentence_transformers import SentenceTransformer
        # Here we load the multilingual CLIP model. Note, this model can only encode text.
        # If you need embeddings for images, you must load the 'clip-ViT-B-32' model
        se_lang_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        se_image_model = SentenceTransformer('clip-ViT-B-32')
        lang_model = lambda queries: se_lang_model.encode(queries, convert_to_tensor=True, show_progress_bar=False).cpu().detach().numpy()
        image_model = lambda images: se_image_model.encode(images, batch_size=1024, convert_to_tensor=True, show_progress_bar=False).cpu().detach().numpy()
    elif MODEL_TYPE == 'clip_italian':
        # import jax
        # from jax import numpy as jnp
        TOKENIZER_NAME = "dbmdz/bert-base-italian-xxl-uncased"
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=None, use_fast=True)
        model = FlaxHybridCLIP.from_pretrained("clip-italian/clip-italian")
        def tokenize(texts):
            inputs = tokenizer(texts, max_length=96, padding="max_length", return_tensors="np")
            return inputs['input_ids'], inputs['attention_mask']

        lang_model = lambda queries: np.asarray(model.get_text_features(*tokenize(queries)))
        image_model = lambda images: np.asarray(model.get_image_features(images.permute(0, 2, 3, 1).numpy(),))
    
    return image_model, lang_model 
