import os
from tqdm import tqdm
import torch

from transformers import logging

from tclip.CLIP import CLIPModel
from tclip.clip_ops import AverageMeter, evaluate_classification#, get_lr
from tclip.inference import get_image_embeddings, find_matches
from tclip.dataset import load_data, prepare_dataframe, build_loaders
from models.ops import get_tokenizer
import configs

import argparse

logging.set_verbosity_warning()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Training clip')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
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



def train_epoch(model, tokenizier, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AverageMeter()
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

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        torch.cuda.empty_cache()
    lr_scheduler.step()
    return loss_meter


def valid_epoch(model, tokenizer, valid_loader):
    loss_meter = AverageMeter()
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

def validate(model, tokenizer, params):
    _, valid_loader = load_data(params)
    with torch.no_grad():
        valid_loss = valid_epoch(model, tokenizer, valid_loader)
    print("val_loss: {:.4f}".format(valid_loss.avg))

def train(model, tokenizer, params):
    logf = open(f'../results/logs/{params.data}_{params.lang}.out', 'w')
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
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
            torch.save(model.state_dict(), params.model_path)
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
    params.image_path = f"{params.data_dir}/{params.data}/images/{configs.image_folders[params.lang]}"
    params.captions_path = f"{params.data_dir}/{params.data}/captions/{params.lang}/{configs.caption_names[params.lang]}"
    params.image_prefix = configs.image_prefixes[params.lang]
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
    params.model_path = f"../results/clips/{params.data}/best_{params.lang}.pt"
    tokenizer = get_tokenizer(params.lang, params.model_name) # will be trained?

    model = CLIPModel(params.lang, params.model_name, not params.resume, temperature=params.temperature).to(params.device)
    if params.resume:
        model.load_state_dict(torch.load(params.model_path))

    if params.is_train:
        train(model, tokenizer, params)
    else:
        model.eval()
        validate(model, tokenizer, params)

        # evaluate(model, tokenizer, params)
        # if params.lang == 'en':
        #     query = "a bus sitting next to the building"
        #     # query = "This is the photo of a man and a horse"
        # else:
        #     query = "un autobus seduto accanto a un edificio"
        #     # query = "Una persona sta guidando"
        # inference(query, model, tokenizer, params)
