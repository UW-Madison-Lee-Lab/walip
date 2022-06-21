import torch, sys
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
from funcy import chunks
from sklearn import metrics
from utils.text_loader import load_vocabs
from utils.image_loader import load_image_dataset
from utils.helper import AverageMeter, accuracy
from models.templates import prompts, generate_texts
import configs
from tqdm import tqdm
import numpy as np


def load_label_embs(model, lang, langs, word_data, data_mode, num_prompts):
    vocab = load_vocabs(lang, langs, word_data, data_mode)
    texts = generate_texts(prompts[word_data][lang], vocab, k=num_prompts)
    K = num_prompts
    text_embeddings = []
    for batch_texts in tqdm(chunks(128*K, texts)):
        with torch.no_grad():
            batch_txt_embs = model.encode_text(batch_texts)
            # ensemble
            batch_size = len(batch_texts) // K
            batch_txt_embs = batch_txt_embs.view(batch_size, K, batch_txt_embs.shape[-1])
            batch_txt_embs = batch_txt_embs.mean(dim=1)
            # normalize after averaging
            batch_txt_embs = F.normalize(batch_txt_embs, dim=-1)
            text_embeddings.append(batch_txt_embs)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    return text_embeddings
    
def validate(model, text_embeddings, dataloader, device, logit_scale=1.0):
    text_embeddings = text_embeddings.type(torch.FloatTensor).to(device)
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    top5, top1 = AverageMeter(), AverageMeter()
    for (images, labels) in tqdm_object:
        # print(batch_idx)
        labels = labels.long().to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(images.to(device))
            image_embeddings = image_embeddings.type(torch.FloatTensor).to(device)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.T * logit_scale
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        precs = accuracy(logits, labels, topk=(1, 5))
        top1.update(precs[0].item(), images.size(0))
        top5.update(precs[1].item(), images.size(0))
        tqdm_object.set_postfix(top1_acc=top1.avg)
        torch.cuda.empty_cache()

    print("Classification on", top1.avg, top5.avg)
    return top1.avg, top5.avg




def evaluate_multilabel_classification(image_data, lang, opts):
    loss = torch.nn.MultiLabelSoftMarginLoss()
    model, text_embeddings, dataloader = load_image_and_class(image_data, lang, opts, multilabel=True)
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    model.eval()

    total = 0
    num_examples = 0
    
    gts = {i:[] for i in range(0, configs.num_classes[image_data])}
    preds = {i:[] for i in range(0, configs.num_classes[image_data])}
    with torch.no_grad():
        for (images, labels) in tqdm_object:
            labels = labels.long().to(opts.device)
            images = images.to(opts.device)
            image_features = model.image_encoder(images)
            image_embeddings = model.image_projection(image_features)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            logits = image_embeddings @ text_embeddings.T * np.exp(model.temperature)
            l = loss(logits, labels).item()
            total += l
            num_examples += image_features.size()[0]
            output = torch.sigmoid(logits)
            pred = output.squeeze().data.cpu().numpy()
            gt = labels.squeeze().data.cpu().numpy()
            for label in range(0, configs.num_classes[image_data]):
                gts[label].extend(gt[:,label])
                preds[label].extend(pred[:,label])

    print("Average Multilabel Loss is: {}".format(total/num_examples))
    
    FinalMAPs = []
    for i in range(0, configs.num_classes[image_data]):
        precision, recall, _ = metrics.precision_recall_curve(gts[i], preds[i])
        FinalMAPs.append(metrics.auc(recall, precision))

    # Print AUC for each class
    indices = [i for i in range(configs.num_classes[image_data])]
    indices = sorted(indices, key = lambda x: FinalMAPs[x])
    with open("../results/AUC.txt", "w") as f:
        for idx in indices:
            f.write("{}: {}\n".format(idx, FinalMAPs[idx]))
    
    return FinalMAPs

