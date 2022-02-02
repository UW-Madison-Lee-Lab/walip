import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils.loader import load_vocabs, load_image_dataset, ViTDataset
from utils.helper import AverageMeter, accuracy
from models.templates import prompts, generate_texts
from models.ops import get_tokenizer, load_models
import configs
from tqdm import tqdm
from transformers import ViTFeatureExtractor
import cv2

# class ViTDataset(Dataset):
#     def __init__(self, dataset):
#         super(ViTDataset, self).__init__()
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         img = self.feature_extractor(self.dataset[index][0], return_tensors="pt")
#         return img['pixel_values'][0], self.dataset[index][1]

def evaluate_classification(image_data, lang, opts):
    vit_image_dataset = load_image_dataset(image_data)
    # from torchvision.datasets import CIFAR100
    # image_dataset = CIFAR100('../dataset/', download=True, train=False)
    # vit_image_dataset = ViTDataset(image_dataset)
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
        # text_embeddings = F.normalize(text_embeddings, dim=-1)
    top5, top1 = AverageMeter(), AverageMeter()
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    for (images, labels) in tqdm_object:
        # print(batch_idx)
        labels = labels.long().to(opts.device)
        images = images.to(opts.device)
        image_features = model.image_encoder(images)
        image_embeddings = model.image_projection(image_features)
        # image_embeddings = F.normalize(image_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.T
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        precs = accuracy(logits, labels, topk=(1, 5))
        top1.update(precs[0].item(), images.size(0))
        top5.update(precs[1].item(), images.size(0))

    print("Classification on", image_data, lang, top1.avg, top5.avg)