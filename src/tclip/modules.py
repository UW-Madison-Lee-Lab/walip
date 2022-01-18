import torch
from torch import nn

from transformers import BertForMaskedLM, BertConfig, AutoConfig, AutoModel
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='google/vit-base-patch16-224-in21k', pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            self.model = ViTModel(ViTConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x):
        # inputs = self.feature_extractor(images=x, return_tensors="pt")
        outputs = self.model(x)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class TextEncoder(nn.Module):
    def __init__(self, lang, model_name, pretrained=True, trainable=True):
        super().__init__()
        
        if lang in ['es', 'en']:
            config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
            if pretrained:
                self.model = BertForMaskedLM.from_pretrained(model_name, config=config)
            else:
                self.model = BertForMaskedLM(config)
        elif lang == 'it':
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            if pretrained:
                self.model = AutoModel.from_pretrained(model_name, config=config)
            else:
                self.model = AutoModel.from_config(config)
                
        for p in self.model.parameters():
            p.requires_grad = trainable
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.hidden_states[-1]
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        projection_dim=256,
        dropout=0.4
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

