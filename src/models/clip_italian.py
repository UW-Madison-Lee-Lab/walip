import numpy as np
from transformers import AutoTokenizer
from clip_italian.modeling_hybrid_clip import FlaxHybridCLIP
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"


def get_italian_models():
    TOKENIZER_NAME = "dbmdz/bert-base-italian-xxl-uncased"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=None, use_fast=True)
    model = FlaxHybridCLIP.from_pretrained("clip-italian/clip-italian")
    def tokenize(texts):
        inputs = tokenizer(texts, max_length=96, padding="max_length", return_tensors="np")
        return inputs['input_ids'], inputs['attention_mask']

    lang_model = lambda queries: np.asarray(model.get_text_features(*tokenize(queries)))
    image_model = lambda images: np.asarray(model.get_image_features(images.permute(0, 2, 3, 1).numpy(),))
    return image_model, lang_model 