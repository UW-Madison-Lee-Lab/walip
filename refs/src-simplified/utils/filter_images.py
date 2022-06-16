
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from models.ops import load_models
from tclip.clip_ops import load_label_embs
from utils.image_loader import load_image_dataset
from utils.helper import AverageMeter, accuracy, generate_path
import configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_correct_images(lang, opts):
    nclasses = configs.num_classes[opts.image_data]
    model, logit_scale, preprocess = load_models(lang, configs.model_names[lang], 'coco', opts.device, opts.large_model)
    label_embeddings = load_label_embs(model, lang, opts.word_data, opts.data_mode, opts.num_prompts)
    image_dataset = load_image_dataset(opts.image_data, preprocess=preprocess)
    dataloader = DataLoader(image_dataset, batch_size=opts.batch_size, shuffle=False, drop_last=True, num_workers=4)
    tqdm_object = tqdm(dataloader, total=len(dataloader))

    indices = {}
    s, total, batch_idx = 0, 0, 0
    top5, top1 = AverageMeter(), AverageMeter()
    for (images, labels) in tqdm_object:
        targets = labels.long().to(opts.device)
        with torch.no_grad():
            image_features = model.encode_image(images.to(opts.device))
            # image_features = F.normalize(image_features, dim=1)
        logits = image_features @ label_embeddings.t() * logit_scale
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))[0].cpu().numpy()

        precs = accuracy(logits, targets, topk=(1, 5))
        top1.update(precs[0].item(), images.size(0))
        top5.update(precs[1].item(), images.size(0))
        inds = np.where(correct == True)[0]
        for x in inds:
            t = labels[x].item()
            if t not in indices:
                indices[t] = []
            indices[t].append(x + batch_idx * images.size(0))
        batch_idx += 1

    fpath = generate_path('img_index', {'image_data': opts.image_data, 'lang': lang, 'num_prompts': opts.num_prompts})
    np.save(fpath, indices)
    print(top1.avg, top5.avg)
    print('#-classes', len(indices))


def find_interesection(data_name, opts):
    def intersect(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    indices = []
    for l in [opts.src_lang, opts.tgt_lang]:
        fpath = generate_path('img_index', {'image_data': opts.image_data, 'lang': l, 'num_prompts': opts.num_prompts})
        indices.append(np.load(fpath, allow_pickle=True).item())

    keys = intersect(list(indices[0].keys()), list(indices[1].keys()))
    ans, lst = {}, []
    for k in sorted(keys):
        values = intersect(indices[0][k], indices[1][k])
        if len(values) > 0:
            ans[k] = values
        else:
            lst.append(k)

    fpath = generate_path('img_shared_index', {'image_data': opts.image_data, 'src_lang': opts.src_lang, 'tgt_lang': opts.tgt_lang, 'num_prompts': opts.num_prompts})
    np.save(fpath, ans)
    print('Not shared: ', lst)