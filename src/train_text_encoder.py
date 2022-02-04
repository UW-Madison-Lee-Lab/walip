import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import logging

from tclip.CLIP import CLIPModel, MyCLIPModel, NewCLIPModel, TextCLIPModel
from tclip.clip_ops import AverageMeter, evaluate_classification, validate, load_image_and_class
from tclip.inference import get_image_embeddings, find_matches
from tclip.dataset import load_data, prepare_dataframe, build_loaders
from models.ops import load_models
import configs

import argparse

logging.set_verbosity_warning()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Training clip')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.001)
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


def encode_image(image_encoder, images):
    image_embeddings = image_encoder(images)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    return image_embeddings    

def train_epoch(image_encoder, text_encoder, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AverageMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    text_encoder.train()
    for batch in tqdm_object:
        images = batch['image'].to(params.device)
        texts = batch['caption']
        with torch.no_grad():
            image_embeddings = encode_image(image_encoder, images)
        text_embeddings = text_encoder.encode_text(texts)
        loss = text_encoder(image_embeddings, text_embeddings)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg)

        torch.cuda.empty_cache()
    lr_scheduler.step()
    return loss_meter


def valid_epoch(image_encoder, text_encoder, valid_loader):
    loss_meter = AverageMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        images = batch['image'].to(params.device)
        texts = batch['caption']
        image_embeddings = encode_image(image_encoder, images)
        text_embeddings = text_encoder.encode_text(texts)
        loss = text_encoder(image_embeddings, text_embeddings)
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


def train(image_encoder, text_encoder, preprocess, params):
    logf = open(f'../results/logs/finetune_{params.data}_{params.lang}.out', 'w')
    print("Training model")
    train_loader, valid_loader = load_data(params)

    # text_embeddings, dataloader = load_image_and_class(model, preprocess, image_data, lang, opts)
    
    # optimizer = torch.optim.SGD(model.mapping.parameters(), params.lr)
    optimizer = torch.optim.AdamW(
        text_encoder.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(params.epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss = train_epoch(image_encoder, text_encoder, train_loader, optimizer, lr_scheduler, step)
        text_encoder.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(image_encoder, text_encoder, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(text_encoder.state_dict(), params.model_path)
            log_write(logf, "Saved Best Model!")
        
        log_write(logf, "epoch {} train_loss: {:.4f} val_loss: {:.4f}".format(epoch, train_loss.avg, valid_loss.avg))

    



if __name__ == "__main__":
    print("Model on " + params.lang)
    params.model_path = f"../results/clips/{params.data}/textencoder_{params.lang}.pt"

    model_name = configs.model_names[params.lang]
    teacher_model, logit_scale, preprocess = load_models('en', model_name, clip_data='coco', device='cuda', large_model=True)

    # teacher_model.eval()
    image_encoder = lambda queries: teacher_model.encode_image(queries)
    text_encoder = TextCLIPModel(params.lang, model_name, logit_scale).to(params.device)

    if params.resume:
        text_encoder.load_state_dict(torch.load(params.model_path))

    if params.is_train:
        train(image_encoder, text_encoder, preprocess, params)
    else:
        params.data_mode = 'test'
        params.word_data = 'cifar10'
        params.image_data = 'cifar10'
        params.txt_dir = f'../dicts/texts/{params.word_data}/'
        params.num_prompts = 1
        text_encoder.eval()
        model = MyCLIPModel(image_encoder, text_encoder, params.device)
        evaluate_classification(params.image_data, params.lang, params, model)
