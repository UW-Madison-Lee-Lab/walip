
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from utils.loader import load_vocabs, load_image_dataset
from models.ops import get_tokenizer, load_models
from models.templates import prompts, generate_texts
from tqdm import tqdm
import torch.nn.functional as F
from utils.helper import AverageMeter, accuracy

import configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_correct_images(lang, opts):
    nclasses = configs.num_classes[opts.image_data]
    bs = configs.image_batchsizes[opts.image_data]

    vocab = load_vocabs(opts, lang)	#CIFAR10
    texts = generate_texts(prompts[lang], vocab, k=opts.num_prompts)
    model_name = configs.model_names[lang]

    model, logit_scale = load_models(lang, model_name, 'coco', opts.device)
    tokenizer = get_tokenizer(lang, model_name)
    text_tokens = tokenizer(texts, padding=True, truncation=True,max_length=200)
    item = {key: torch.tensor(values).to(opts.device) for key, values in text_tokens.items()}
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        text_features = model.text_projection(text_features)

    image_dataset = load_image_dataset(opts.image_data)
    dataloader = DataLoader(image_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)
    tqdm_object = tqdm(dataloader, total=len(dataloader))

    indices = {}
    s, total, batch_idx = 0, 0, 0
    top5, top1 = AverageMeter(), AverageMeter()
    for (images, labels) in tqdm_object:
        targets = labels.long().cuda()
        with torch.no_grad():
            image_features = model.image_encoder(images.cuda()).float()
            image_features = model.image_projection(image_features)
            # if is_eng_clip:
            #     image_features = image_model.encode_image(images.cuda()).float()
            # else:
            #     image_features = image_model(images)
            #     image_features = torch.from_numpy(image_features).cuda()
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ text_features.t()
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
            indices[t].append(x + batch_idx * bs)
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