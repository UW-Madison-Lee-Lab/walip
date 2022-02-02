import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loader import load_vocabs, load_image_dataset, ViTDataset
from utils.helper import AverageMeter, accuracy
from models.templates import prompts, generate_texts
from models.ops import get_tokenizer, load_models
import configs
from tqdm import tqdm
from sklearn import metrics



def evaluate_classification(image_data, lang, opts):
    vit_image_dataset = load_image_dataset(image_data)
    dataloader = DataLoader(vit_image_dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=4)
    tqdm_object = tqdm(dataloader, total=len(dataloader))

    vocab = load_vocabs(opts, lang)
    texts = generate_texts(prompts[lang], vocab, k=opts.num_prompts)

    model_name = configs.model_names[lang]
    tokenizer = get_tokenizer(lang, model_name)
    model, logit_scale = load_models(lang, model_name, 'coco', opts.device) 
    # model = model.to(opts.device)   

    text_tokens = tokenizer(texts, padding=True, truncation=True,max_length=200)
    item = {key: torch.tensor(values).to(opts.device) for key, values in text_tokens.items()}
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
    top5, top1 = AverageMeter(), AverageMeter()
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    for (images, labels) in tqdm_object:
        # print(batch_idx)
        labels = labels.long().to(opts.device)
        images = images.to(opts.device)
        image_features = model.image_encoder(images)
        image_embeddings = model.image_projection(image_features)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.T
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        precs = accuracy(logits, labels, topk=(1, 5))
        top1.update(precs[0].item(), images.size(0))
        top5.update(precs[1].item(), images.size(0))

    print("Classification on", image_data, lang, top1.avg, top5.avg)



def evaluate_multiclass_classification(model, tokenizer, params, data_loader):
    loss = torch.nn.MultiLabelSoftMarginLoss()
    words = []
    # Load text embeddings for each possible image class
    with open('../../dicts/texts/coco/coco_en_labels.txt') as f:
        lines = f.readlines()
        texts = []
        for i, desc in enumerate(lines):
            desc = desc.strip().lower()
            words.append(desc)
            if params.lang == 'en':
                texts.append("A photo of a {}.".format(desc))
            elif params.lang == 'es':
                texts.append("una foto de un {}.".format(desc))
            else:
                texts.append("una foto di un {}.".format(desc))

        
        text_tokens = tokenizer(texts, padding=True, truncation=True,max_length=params.max_length)
        item = {key: torch.tensor(values).to(params.device) for key, values in text_tokens.items()}
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=item["input_ids"], attention_mask=item["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)

            # Not sure whether to normalize:
            #text_embeddings = F.normalize(text_embeddings, dim=-1)


    model.eval()
    tqdm_object = tqdm(data_loader, total=len(data_loader))
    total = 0
    num_batches = 0
    
    gts = {i:[] for i in range(0, 80)}
    preds = {i:[] for i in range(0, 80)}
    with torch.no_grad():
        for batch in tqdm_object:

            labels = batch["labels"].to(params.device)
            images = batch["image"].to(params.device)
            
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