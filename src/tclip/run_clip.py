import os
import gc
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

from CLIP import CLIPModel
from transformers import BertTokenizer, AutoTokenizer

from clip_ops import AvgMeter, evaluate_classification, get_lr
from inference import get_image_embeddings, find_matches
from dataset import load_data, prepare_dataframe, build_loaders

import argparse
from transformers import logging


logging.set_verbosity_warning()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Training clip')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--factor", type=float, default=0.5)

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--num_projection_layers", type=int, default=1)
parser.add_argument("--projection_dim", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.07)

parser.add_argument("--image_embedding", type=int, default=768)
parser.add_argument("--text_embedding", type=int, default=768)
parser.add_argument("--max_length", type=int, default=200)

parser.add_argument("--text_encoder_model", type=str, default="distilbert-base-uncased")
parser.add_argument("--text_tokenizer", type=str, default="distilbert-base-uncased")

parser.add_argument("--lang", type=str, default='en', help="Source language")
parser.add_argument("--data", type=str, default='coco', help="Source language")

parser.add_argument("--resume", action='store_true')
parser.add_argument("--is_train", action='store_true')


# parse parameters
params = parser.parse_args()

model_names = {
    "en": "bert-base-uncased",
    "es": "../pretrained/uncased/",
    "it": "dbmdz/bert-base-italian-uncased"
}

image_folders = {
    "en": "val2017",
    "es": "spanish_images",
    "it": "val2014",
}

image_prefixes = {
    'en': '',
    'es': '',
    'it': 'COCO_val2014_'
}
caption_names = {
    "en": "processed_captions_val2017.csv",
    "es": "es/",
    "it": "processed_captions_val2014.csv",
}

data_folder = "../../data"
params.model_name = model_names[params.lang]
params.image_path = f"{data_folder}/{params.data}/images/{image_folders[params.lang]}"
params.captions_path = f"{data_folder}/{params.data}/captions/{params.lang}/{caption_names[params.lang]}"
params.image_prefix = image_prefixes[params.lang]

params.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_tokenizer(lang, model_name):
    if lang in ['es', 'en']:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    elif lang == 'it':
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    return tokenizer


def train_epoch(model, tokenizier, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        text_tokens = tokenizer(batch['caption'], padding=True, truncation=True,max_length=params.max_length)
        batch['image'] = batch['image'].to(params.device)
        for k, v in text_tokens.items():
            batch[k] = torch.tensor(v).to(params.device)
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if step == "batch":

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        torch.cuda.empty_cache()
    lr_scheduler.step()
    return loss_meter

def valid_epoch(model, tokenizer, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        text_tokens = tokenizer(batch['caption'], padding=True, truncation=True,max_length=params.max_length)
        batch['image'] = batch['image'].to(params.device)
        for k, v in text_tokens.items():
            batch[k] = torch.tensor(v).to(params.device)
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def log_write(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)

def train(model, tokenizer, params):
    logf = open(f'../../results/logs/{params.data}_{params.lang}.out', 'w')
    print("Training model")
    train_loader, valid_loader = load_data(params)
    # optimizer = torch.optim.AdamW(
        # model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    # )
    optimizer = torch.optim.SGD(model.parameters(), params.lr)
    
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=params.patience, factor=params.factor
    # )
    # lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.95 ** epoch
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0002, max_lr=0.002,step_size_up=5,mode="triangular")
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(params.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, tokenizer, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), model_path)
            log_write(logf, "Saved Best Model!")
        
        log_write(logf, "epoch {} train_loss: {:.4f} val_loss: {:.4f}".format(epoch, train_loss.avg, valid_loss.avg))

def evaluate(model, tokenizer, params):
    model.eval()
    print("Evaluating classification!")
    evaluate_classification(model, tokenizer, params)


def inference(query, model, tokenizer, params):
    model.eval()
    print("Evaluate model on " + params.lang)
    orig_lang = params.lang

    # image-path: eng
    params.lang = 'en'
    params.image_path = f"{data_folder}/{params.data}/images/{image_folders[params.lang]}"
    params.captions_path = f"{data_folder}/{params.data}/captions/{params.lang}/{caption_names[params.lang]}"
    params.image_prefix = image_prefixes[params.lang]
    # train_loader, valid_loader = load_data(params)
    _, valid_df = prepare_dataframe('en', params.captions_path)
    valid_loader = build_loaders(valid_df, "valid", params)
    image_embeddings = get_image_embeddings(model, valid_loader, params.device)

    image_ids = valid_df["image_id"].values
    image_filenames = [f"{params.image_path}/{params.image_prefix}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 


    # set-back lang
    params.lang = orig_lang
    find_matches(model, 
        tokenizer,
        image_embeddings,
        query=query,
        image_filenames=image_filenames,
        lang=params.lang,
        n=9, device=params.device)



if __name__ == "__main__":
    print("Model on " + params.lang)
    model_path = f"../../results/clips/{params.data}/best_{params.lang}.pt"
    tokenizer = get_tokenizer(params.lang, params.model_name) # will be trained?
    if params.resume:
        model = CLIPModel(params.lang, params.model_name, pretrained=False, temperature=params.temperature).to(params.device)
        model.load_state_dict(torch.load(model_path))
    else:
        model = CLIPModel(params.lang, params.model_name, pretrained=True, temperature=params.temperature).to(params.device)

    if params.is_train:
        train(model, tokenizer, params)
    else:
        # evaluate(model, tokenizer, params)
        if params.lang == 'en':
            # query = "a bus sitting next to the building"
            query = "A man riding a horse"
        else:
            # query = "un autobus seduto accanto a un edificio"
            query = "Una persona sta guidando"
        inference(query, model, tokenizer, params)
