import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from modules import ImageEncoder_resnet, ImageEncoder_ViT, TextEncoder, ProjectionHead



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
        text_embedding=768,
        max_length=200
    ):
        super().__init__()
        projection_dim = 256
        if lang == 'it':
            image_embedding = 2048
            self.image_encoder = ImageEncoder_resnet(pretrained=pretrained)
        else:
            image_embedding=768
            self.image_encoder = ImageEncoder_ViT(pretrained=pretrained)
        self.text_encoder = TextEncoder(lang, model_name, pretrained=pretrained)
        # self.image_projection = nn.Linear(image_embedding, projection_dim)
        # self.text_projection = nn.Linear(text_embedding, projection_dim)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.max_length = max_length
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    def get_embeddings(self, batch):
        # Getting Image and Text Features
        with torch.no_grad():
            image_features = self.image_encoder(batch["image"])
            text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        # image_embeddings = F.normalize(self.image_projection(image_features), dim=1)
        # text_embeddings = F.normalize(self.text_projection(text_features), dim=1)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings
    
    def multilabel_classify(self, batch, text_features):
        image_features = self.image_encoder(batch["image"])
        image_embeddings = self.image_projection(image_features)
        return image_embeddings @ text_features.T * np.exp(self.temperature)

    def classify(self, batch):
        image_embeddings, text_embeddings = self.get_embeddings(batch)
        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        return logits

    def forward_v0(self, batch):
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


