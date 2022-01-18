import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]





def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_classification(model, tokenizer, params):

    preprocess = transforms.Compose([
        Resize([params.img_size], interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
    ])
    image_dataset = CIFAR10('../../data', transform=preprocess, download=True, train=False)
    dataloader = DataLoader(image_dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=4)

    # texts = [f"This is a photo of a " + desc for desc in image_dataset.classes]
    with open('../../dicts/texts/cifar10/cifar10_en_test.txt') as f:
        lines = f.readlines()
    texts = []
    for desc in lines:
        desc = desc.strip().lower()
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
        text_embeddings = F.normalize(text_embeddings, dim=-1)

    top5, top1 = AvgMeter(), AvgMeter()
    indices = {}
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # print(batch_idx)
        labels = labels.long().to(params.device)
        images = inputs.to(params.device)
        image_features = model.image_encoder(images)
        image_embeddings = model.image_projection(image_features)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.T
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        precs = accuracy(logits, labels, topk=(1, 5))
        top1.update(precs[0].item(), inputs.size(0))
        top5.update(precs[1].item(), inputs.size(0))

    print(top1.avg, top5.avg)