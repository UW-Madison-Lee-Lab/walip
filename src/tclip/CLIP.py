import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from modules import ImageEncoder, TextEncoder, ProjectionHead



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
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.text_encoder = TextEncoder(lang, model_name, pretrained=pretrained)
        self.image_projection = nn.Linear(image_embedding, projection_dim)
        self.text_projection = nn.Linear(text_embedding, projection_dim)
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
        image_embeddings = F.normalize(self.image_projection(image_features), dim=1)
        text_embeddings = F.normalize(self.text_projection(text_features), dim=1)
        return image_embeddings, text_embeddings
    
    def classify(self, batch):
        image_embeddings, text_embeddings = self.get_embeddings(batch)
        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        return logits

    def forward(self, batch):
        image_embeddings, text_embeddings = self.get_embeddings(batch)
        # Calculating the Loss
        logits = (image_embeddings @ text_embeddings.T) * np.exp(self.temperature)
        targets = torch.tensor(np.arange(logits.shape[0])).to(logits.device)
        texts_loss = self.criterion(logits.T, targets)
        images_loss = self.criterion(logits, targets)
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


