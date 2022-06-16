
import numpy as np
from transformers import AutoTokenizer
from clip_italian.modeling_hybrid_clip import FlaxHybridCLIP
import os
import numpy as np

import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from templates import generate_texts, templates
from helper import load_vocabs

os.environ["TOKENIZERS_PARALLELISM"] = "false"


TOKENIZER_NAME = "dbmdz/bert-base-italian-xxl-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=None, use_fast=True)
model = FlaxHybridCLIP.from_pretrained("clip-italian/clip-italian")
def tokenize(texts):
    inputs = tokenizer(texts, max_length=96, padding="max_length", return_tensors="np")
    return inputs['input_ids'], inputs['attention_mask']

language_model = lambda queries: np.asarray(model.get_text_features(*tokenize(queries)))
image_model = lambda images: np.asarray(model.get_image_features(images.permute(0, 2, 3, 1).numpy(),))


val_preprocess = transforms.Compose([
    Resize([224], interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

nclasses = {'imagenet': 1000, 'cifar100': 100}

def evaluate(data_name, language='it'):
    vocabs = load_vocabs(data_name, language)	
    texts = generate_texts(templates[language], vocabs) 

    class_embeddings = language_model(texts)
    class_embeddings = class_embeddings / np.linalg.norm(class_embeddings, axis=-1, keepdims=True)
    K = len(templates[language])
    B = class_embeddings.shape[0] // K
    class_embeddings = class_embeddings.reshape(B, K, -1)
    class_embedding = np.mean(class_embeddings, axis=1) # for different vector
    class_embedding /= np.linalg.norm(class_embedding, axis=-1, keepdims=True)
    text_features = class_embedding

    if data_name == 'cifar100':
        bs = 512
        image_dataset = CIFAR100('../data', transform=val_preprocess, download=True, train=False)
        dataloader = DataLoader(image_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)
    else:
        bs = 32
        data_dir = '../../repgan/data/imagenet/val'
        dataset = ImageFolder(data_dir, val_preprocess)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)

    indices = {}
    c = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        image_features = image_model(inputs)
        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
        logits = torch.Tensor(image_features) @ torch.Tensor(text_features).t()
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(Variable(labels).view(1, -1).expand_as(pred))
        correct = correct[0]
        c += correct.sum()
        total += bs
        print('Accuracy: {:.4f}'.format(c/total))
        inds = np.where(correct.cpu().numpy() == True)[0]
        for x in inds:
            t = labels[x].item()
            if t not in indices:
                indices[t] = []
            indices[t].append(x + batch_idx * bs)
        
    print('Final Accuracy: {:.4f}'.format(c/t))
    fname = '../dicts/{}_{}_correct_index'.format(data_name, language)
    np.save(fname, indices)
    f = open(fname + ".txt", 'w')
    for i in range(nclasses[data_name]):
        if i in indices:
            f.write("{} {}\n".format(i, indices[i]))
        else:
            print(i, 'not in indices')
    f.close()



if __name__ == '__main__':
    evaluate('cifar100', 'it')



