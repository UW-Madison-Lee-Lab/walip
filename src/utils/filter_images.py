
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from models.ops import load_models
from tclip.clip_ops import load_image_and_class
from tqdm import tqdm
import torch.nn.functional as F
from utils.helper import AverageMeter, accuracy

import configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_correct_images(lang, opts):
    nclasses = configs.num_classes[opts.image_data]
    model_name = configs.model_names[lang]
    model, logit_scale, preprocess = load_models(lang, model_name, 'coco', opts.device, opts.large_model)
    text_features, dataloader = load_image_and_class(model, preprocess, opts.image_data, lang, opts)
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    indices = {}
    s, total, batch_idx = 0, 0, 0
    top5, top1 = AverageMeter(), AverageMeter()
    for (images, labels) in tqdm_object:
        targets = labels.long().to(opts.device)
        with torch.no_grad():
            image_features = model.encode_image(images.to(opts.device))
        logits = image_features @ text_features.t() * logit_scale
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct = correct[0]
        s += correct.sum()
        total += targets.shape[0]

        precs = accuracy(logits, targets, topk=(1, 5))
        top1.update(precs[0].item(), images.size(0))
        top5.update(precs[1].item(), images.size(0))
        inds = np.where(correct.cpu().numpy() == True)[0]
        for x in inds:
            t = labels[x].item()
            if t not in indices:
                indices[t] = []
            indices[t].append(x + batch_idx * opts.batch_size)
        batch_idx += 1

    fname = os.path.join(opts.img_dir, f'{opts.image_data}_{lang}_correct_index')
    np.save(fname, indices)
    f = open(fname + ".txt", 'w')
    for i in range(nclasses):
        if i in indices:
            f.write("{} {}\n".format(i, indices[i]))
        else:
            print(i, 'not in indices')
    f.close()

    print('Classification accuracy: {:.4f}'.format(s/total))
    print(top1.avg, top5.avg)


def find_interesection(data_name, opts):
    def intersect(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    indices = []
    for l in [opts.src_lang, opts.tgt_lang]:
        fpath = os.path.join(opts.img_dir, f'{data_name}_{l}_correct_index.npy')
        indices.append(np.load(fpath, allow_pickle=True).item())

    keys = intersect(list(indices[0].keys()), list(indices[1].keys()))
    ans = {}
    for k in sorted(keys):
        values = intersect(indices[0][k], indices[1][k])
        if len(values) > 0:
            ans[k] = values[0]
        else:
            print(k) # 36
    fname = os.path.join(opts.img_dir, '{}_{}_{}_index'.format(data_name, opts.src_lang, opts.tgt_lang))
    np.save(fname, ans)
    f = open(fname + ".txt", 'w')
    # image_dataset.classes[0]
    for k in sorted(ans.keys()):
        f.write("{} {}\n".format(k, ans[k]))
    f.close()