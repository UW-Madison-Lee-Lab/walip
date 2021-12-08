
import os
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.loader import load_vocabs, load_image_dataset
from models.ops import get_batch_clip_based_text_features, load_models
from models.prompt_templates import prompts, generate_texts

import configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

image_dir = '../dicts/images/'

def find_correct_images(lang, image_name):
    nclasses = configs.num_classes[image_name]
    bs = {'imagenet':32, 'cifar100': 128, 'cifar10':128}[image_name]

    vocab = load_vocabs(image_name, lang, mode='full')	
    template_size = configs.num_prompts
    texts = generate_texts(prompts[lang], vocab, k=template_size)
    is_eng_clip, image_model, text_model, logit_scale = load_models(lang)
    text_features = get_batch_clip_based_text_features(text_params=(template_size, texts), model_params=(is_eng_clip, text_model))
    text_features = text_features.cuda()

    image_dataset = load_image_dataset(image_name)
    dataloader = DataLoader(image_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)
    indices = {}
    s = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        targets = Variable(labels).long().cuda()
        images = Variable(inputs)
        with torch.no_grad():
            if is_eng_clip:
                image_features = image_model.encode_image(images.cuda()).float()
            else:
                image_features = image_model(images)
                image_features = torch.from_numpy(image_features).cuda()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.t()
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct = correct[0]
        s += correct.sum()
        total += inputs.shape[0]
        inds = np.where(correct.cpu().numpy() == True)[0]
        for x in inds:
            t = labels[x].item()
            if t not in indices:
                indices[t] = []
            indices[t].append(x + batch_idx * bs)
        

    fname = os.path.join(image_dir, '{}_{}_correct_index'.format(image_name, lang))
    np.save(fname, indices)
    f = open(fname + ".txt", 'w')
    for i in range(nclasses):
        if i in indices:
            f.write("{} {}\n".format(i, indices[i]))
        else:
            print(i, 'not in indices')
    f.close()

    print('Classification accuracy: {:.4f}'.format(s/total))


def find_interesection(data_name, langs=['en', 'it']):
    def intersect(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    indices = []
    for l in langs:
        fpath = os.path.join(image_dir, '{}_{}_correct_index.npy'.format(data_name, l))
        indices.append(np.load(fpath, allow_pickle=True).item())

    keys = intersect(list(indices[0].keys()), list(indices[1].keys()))
    ans = {}
    for k in sorted(keys):
        values = intersect(indices[0][k], indices[1][k])
        if len(values) > 0:
            ans[k] = values[0]
        else:
            print(k) # 36
    fname = os.path.join(image_dir, '{}_{}_{}_index'.format(data_name, langs[0], langs[1]))
    np.save(fname, ans)
    f = open(fname + ".txt", 'w')
    # image_dataset.classes[0]
    for k in sorted(ans.keys()):
        f.write("{} {}\n".format(k, ans[k]))
    f.close()