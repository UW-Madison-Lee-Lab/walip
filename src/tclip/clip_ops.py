import torch, sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.text_loader import load_vocabs
from utils.image_loader import load_image_dataset
from utils.helper import AverageMeter, accuracy
from models.templates import prompts, generate_texts
from models.ops import load_models
import configs
from tqdm import tqdm


def load_image_and_class(model, preprocess, image_data, lang, opts):
    vocab = load_vocabs(opts, lang)
    texts = generate_texts(prompts[lang], vocab, k=opts.num_prompts)

    with torch.no_grad():
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings.view(-1, opts.num_prompts, text_embeddings.shape[-1])
        text_embeddings = text_embeddings.mean(dim=1)

    image_dataset = load_image_dataset(image_data, preprocess=preprocess)
    dataloader = DataLoader(image_dataset, batch_size=opts.batch_size, shuffle=False, drop_last=True, num_workers=4)

    return text_embeddings, dataloader


def evaluate_classification(image_data, lang, opts, model=None):
    if model is None:
        model_name = configs.model_names[lang]
        model, logit_scale, preprocess = load_models(lang, model_name, 'coco', opts.device, opts.large_model) 
    else:
        preprocess = None
    text_embeddings, dataloader = load_image_and_class(model, preprocess, image_data, lang, opts)
    validate(model, text_embeddings, dataloader, opts.device)

def validate(model, text_embeddings, dataloader, device):
    text_embeddings = text_embeddings.type(torch.cuda.FloatTensor)
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    top5, top1 = AverageMeter(), AverageMeter()
    for (images, labels) in tqdm_object:
        # print(batch_idx)
        labels = labels.long().to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(images.to(device))
            image_embeddings = image_embeddings.type(torch.cuda.FloatTensor)
        logits = image_embeddings @ text_embeddings.T
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        precs = accuracy(logits, labels, topk=(1, 5))
        top1.update(precs[0].item(), images.size(0))
        top5.update(precs[1].item(), images.size(0))
        # sys.stdout.write("Top1 {:.4f} Top5: {:.4f}".format(top1.avg, top5.avg)) 
        # sys.stdout.flush() 
        tqdm_object.set_postfix(top1_acc=top1.avg)
        torch.cuda.empty_cache()

    print("Classification on", top1.avg, top5.avg)




def evaluate_multiclass_classification(image_data, lang, opts):
    loss = torch.nn.MultiLabelSoftMarginLoss()
    words = []

    model, text_embeddings, dataloader = load_image_and_class(image_data, lang, opts)
    tqdm_object = tqdm(dataloader, total=len(dataloader))
    model.eval()

    total = 0
    num_batches = 0
    
    gts = {i:[] for i in range(0, 80)}
    preds = {i:[] for i in range(0, 80)}
    with torch.no_grad():
        for batch in tqdm_object:
            labels = batch["labels"].to(opts.device)
            images = batch["image"].to(opts.device)
            
            # Should be of shape (batch_sz, num_classes)
            logits = model.multilabel_classify(images, text_embeddings)
            l = loss(logits, labels).item()
            total += l
            num_batches += 1
            output = torch.sigmoid(logits)
            pred = output.squeeze().data.cpu().numpy()
            gt = labels.squeeze().data.cpu().numpy()

            for label in range(0, 80):
                gts[label].extend(gt[:,label])
                preds[label].extend(pred[:,label])

    print("Average Multilabel Loss is: {}".format(total/num_batches))
    
    FinalMAPs = []
    for i in range(0, 80):
        precision, recall, _ = metrics.precision_recall_curve(gts[i], preds[i])
        FinalMAPs.append(metrics.auc(recall, precision))

    # Print AUC for each class
    indices = [i for i in range(80)]
    indices = sorted(indices, key = lambda x: FinalMAPs[i])
    with open("../../results/AUC.txt", "w") as f:
        for idx in indices:
            f.write("{}: {}\n".format(words[idx], FinalMAPs[idx]))


    
    return FinalMAPs