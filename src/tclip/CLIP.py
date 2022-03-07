import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tclip.modules import ImageEncoder_resnet, ImageEncoder_ViT, TextEncoder, ProjectionHead
from transformers import BertTokenizer, AutoTokenizer

def get_tokenizer(lang, model_name):
    if lang == 'en':
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    elif lang in ['es', 'it']:
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    return tokenizer

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



class CLIPModel(nn.Module):
    def __init__(
        self,
        lang,
        model_name,
        pretrained=True,
        temperature=0.07,
        image_embedding=768,
        text_embedding=512,
        max_length=200,
        device='cuda'
    ):
        super().__init__()
        projection_dim = 256
        # if lang == 'it':
        #     image_embedding = 2048
        #     self.image_encoder = ImageEncoder_resnet(pretrained=pretrained)
        # else:
        image_embedding=768
        self.lang = lang
        if self.lang == 'es':
            from transformers import CLIPTextModel, CLIPVisionModel
            self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        else: 
            self.image_encoder = ImageEncoder_ViT(pretrained=pretrained)
            # self.image_encoder = ImageEncoder_ViT(pretrained=pretrained)
            self.text_encoder = TextEncoder(lang, model_name, pretrained=pretrained)
        # self.image_projection = nn.Linear(image_embedding, projection_dim)
        # self.text_projection = nn.Linear(text_embedding, projection_dim)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.max_length = max_length
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = device
        self.tokenizer = get_tokenizer(lang, model_name)


    def get_embeddings(self, batch):
        # Getting Image and Text Features
        if self.lang == 'es':
            image_features = self.image_encoder(batch["image"]).last_hidden_state[:, 0, :]
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
        else: #'en'
            image_features = self.image_encoder(batch["image"])
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        # image_features = self.image_encoder(batch["image"])
        # text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # Getting Image and Text Embeddings (with same dimension)
        # image_embeddings = F.normalize(self.image_projection(image_features), dim=1)
        # text_embeddings = F.normalize(self.text_projection(text_features), dim=1)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings
    
    def classify(self, batch):
        image_embeddings, text_embeddings = self.get_embeddings(batch)
        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        return logits

    def forward_v1(self, batch):
        image_embeddings, text_embeddings = self.get_embeddings(batch)
        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        targets = torch.tensor(np.arange(logits.shape[0])).to(logits.device)
        texts_loss = self.criterion(logits.T, targets)
        images_loss = self.criterion(logits, targets)
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def forward(self, batch):
        # Getting Image and Text Features
        image_embeddings, text_embeddings = self.get_embeddings(batch)

        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * np.exp(self.temperature), dim=-1
        )
        texts_loss = cross_entropy(logits.T, targets.T, reduction='mean')
        images_loss = cross_entropy(logits, targets, reduction='mean')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss


    def encode_image(self, images):
        image_features = self.image_encoder(images)
        image_embeddings = self.image_projection(image_features)
        # image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings

    def encode_text(self, texts):
        text_tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=200)
        item = {key: torch.tensor(values).to(self.device) for key, values in text_tokens.items()}
        text_features = self.text_encoder(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)
        # text_embeddings = F.normalize(text_embeddings, dim=-1)
        return text_embeddings



class NewCLIPModel(object):
    def __init__(self, image_encoder, text_encoder, text_projection, mapping, tokenizer, device) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.device = device
        self.temperature = 0.07
    
    def encode_image(self, images):
        image_embeddings = self.image_encoder(images)
        # image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings

    def encode_text(self, texts):
        text_tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=200)
        item = {key: torch.tensor(values).to(self.device) for key, values in text_tokens.items()}
        text_features = self.text_encoder(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)
        text_embeddings = self.mapping(text_embeddings.detach())
        # text_embeddings = F.normalize(text_embeddings, dim=-1)
        return text_embeddings

    def regularizer(self):
        W = self.mapping.weight
        I = torch.eye(W.shape[0]).to(self.device)
        R = W @ W.T - I
        return torch.norm(W, 'fro')

    
    def forward(self, images, texts):
        # Getting Image and Text Features
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(texts)

        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * np.exp(self.temperature), dim=-1
        )
        texts_loss = cross_entropy(logits.T, targets.T, reduction='mean')
        images_loss = cross_entropy(logits, targets, reduction='mean')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss #+ 0.1 * self.regularizer()



class TextCLIPModel(nn.Module):
    def __init__(
        self,
        lang,
        model_name,
        logit_scale=1.0,
        pretrained=True,
        text_embedding=768,
        max_length=200,
        device='cuda'
    ):
        super().__init__()
        self.text_encoder = TextEncoder(lang, model_name, pretrained=pretrained)
        # self.text_projection = nn.Linear(text_embedding, projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding,     projection_dim=512, dropout=0.1)
        self.logit_scale = logit_scale
        self.max_length = max_length
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = device
        self.tokenizer = get_tokenizer(lang, model_name)
    

    def forward(self, image_embeddings, text_embeddings):
        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * self.logit_scale
        targets = torch.tensor(np.arange(logits.shape[0])).to(logits.device)
        texts_loss = self.criterion(logits.T, targets)
        images_loss = self.criterion(logits, targets)
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

   

    def encode_text(self, texts):
        text_tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=200)
        item = {key: torch.tensor(values).to(self.device) for key, values in text_tokens.items()}
        text_features = self.text_encoder(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)
        # text_embeddings = F.normalize(text_embeddings, dim=-1)
        return text_embeddings



class MyCLIPModel(object):
    def __init__(self, image_encoder, text_encoder, device) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.device = device
    
    def encode_image(self, images):
        with torch.no_grad():
            image_embeddings = self.image_encoder(images)
            # image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings

    def encode_text(self, texts):
        return self.text_encoder.encode_text(texts)
