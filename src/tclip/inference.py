import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import date

def get_image_embeddings(model, valid_loader, device):
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(device))
            image_embeddings = model.image_projection(image_features)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

def find_matches(model, tokenizer, image_embeddings, query, image_filenames, n=9, lang='en', device='cuda'):
    text_tokens = tokenizer([query], padding=True, truncation=True,max_length=200)
    item = {key: torch.tensor(values).to(device) for key, values in text_tokens.items()}
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=item["input_ids"], attention_mask=item["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    dot_similarity = text_embeddings @ image_embeddings.T
    
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(match)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.savefig(f'../results/plots/answer_{lang}_{date.today()}.png')
   
