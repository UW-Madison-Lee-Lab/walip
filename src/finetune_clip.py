import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from tclip.clip_ops import AverageMeter, evaluate_classification
from tclip.dataset import load_data
import configs
import clip
from utils.helper import log as log_write
import argparse
from transformers import AutoTokenizer
from typing import Any, Union, List
from tclip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# logging.set_verbosity_warning()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Training clip')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-7)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--num_projection_layers", type=int, default=1)
parser.add_argument("--projection_dim", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.07)

parser.add_argument("--max_length", type=int, default=200)

parser.add_argument("--text_encoder_model", type=str, default="distilbert-base-uncased")
parser.add_argument("--text_tokenizer", type=str, default="distilbert-base-uncased")

parser.add_argument("--lang", type=str, default='it', help="Source language")
parser.add_argument("--data", type=str, default='coco', help="Source language")
parser.add_argument("--data_dir", type=str, default='../dataset', help="Source language")


parser.add_argument("--resume", action='store_true')
parser.add_argument("--is_train", action='store_true')


# parse parameters
params = parser.parse_args()


params.model_name = configs.model_names[params.lang]
params.image_path = f"{params.data_dir}/{params.data}/images/{configs.image_folders[params.lang]}"
params.captions_path = f"{params.data_dir}/{params.data}/captions/{params.lang}/{configs.caption_names[params.lang]}"
params.image_prefix = configs.image_prefixes[params.lang]
params.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


if params.lang == 'it':
    # tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    # tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased", do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    # tokenizer = AutoTokenizer.from_pretrained("idb-ita/gilberto-uncased-from-camembert", do_lower_case=True)

_tokenizer = _Tokenizer()
def italian_tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, _tokenizer=None) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if _tokenizer is None:
        _tokenizer = _Tokenizer()
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def tokenize(texts):
    return clip.tokenize(texts, truncate=True)
    if params.lang == 'en':
        return clip.tokenize(texts)
    else:
        # input_ids = torch.tensor(tokenizer.encode(texts)).unsqueeze(0)  
        # token_list = tokenizer.convert_ids_to_tokens(input_ids) 
        return italian_tokenize(texts)


criterion = torch.nn.CrossEntropyLoss()
def nce_loss(logits_per_image, logits_per_text):
    # targets = torch.tensor(np.arange(logits_per_image.shape[0])).to(params.device)
    ground_truth = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=params.device)
    texts_loss = criterion(logits_per_text, ground_truth)
    images_loss = criterion(logits_per_image, ground_truth)
    loss = (images_loss + texts_loss) / 2.0
    return loss

def valid_epoch(model, valid_loader):
    loss_meter = AverageMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        images = batch['image'].to(params.device)
        texts = batch['caption']
        text_tokens = tokenize(texts).to(params.device)
        logits_per_image, logits_per_text = model(images, text_tokens)
        loss = nce_loss(logits_per_image, logits_per_text)
        loss_meter.update(loss.item(), logits_per_image.shape[0])
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def train(model, params):
    logf = open(f'../results/logs/finetune_wtm_{params.lang}.out', 'w')
    print("Training model")
    train_loader, valid_loader = load_data(params)
    trainable_params = list(model.text_projection) + list(model.visual.proj)
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    # )
    # {'params':img_encoder}, {'params': txt_decoder}]
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.001, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-8,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    # optimizer.param_groups[0]['lr'] = 1e-8
    # optimizer.param_groups[1]['lr'] = 1e-5
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=params.epochs * (2*len(train_loader)-1),
    #                                             base_momentum=0.0, max_momentum=0.5, pct_start=0.1, div_factor=1e2, final_div_factor=1e4)
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    # criterion = TripletLoss(device)
    step = "epoch"
    best_loss = float('inf')
    for epoch in range(params.epochs):
        print(f"Epoch: {epoch + 1}")
        # train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        loss_meter = AverageMeter()
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        model.train()
        
        for batch in tqdm_object:
            images = batch['image'].to(params.device)
            texts = batch['caption']
            text_tokens = tokenize(texts).to(params.device)
            logits_per_image, logits_per_text = model(images, text_tokens)
            optimizer.zero_grad()
            loss = nce_loss(logits_per_image, logits_per_text)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.shape[0])
            tqdm_object.set_postfix(train_loss=loss_meter.avg)

            torch.cuda.empty_cache()
        
        train_loss = loss_meter

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), params.model_path)
            log_write(logf, "Saved Best Model!")
        
        torch.save(model.state_dict(), params.model_last_path)
        log_write(logf, "epoch {} train_loss: {:.4f} val_loss: {:.4f}".format(epoch, train_loss.avg, valid_loss.avg))

        lr_scheduler.step()

    

if __name__ == "__main__":
    print("Model on " + params.lang)
    params.model_path = f"../results/clips/wtm/{params.lang}.pt"
    params.model_last_path = f"../results/clips/wtm/{params.lang}_last.pt"


    model_name = configs.model_names[params.lang]
    if params.resume:
        model, preprocess = clip.load(params.model_path)
    else:
        model, preprocess = clip.load("ViT-B/32")
    model = model.to(params.device)


    if params.is_train:
        train(model, params)
    else:
        params.data_mode = 'test'
        params.word_data = 'cifar10'
        params.image_data = 'cifar10'
        params.txt_dir = f'../dicts/texts/{params.word_data}/'
        params.num_prompts = 1
        model.eval()
        evaluate_classification(params.image_data, params.lang, params, model)
