import os
import cv2
import torch
import albumentations as A
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor

import config as CFG


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.captions = list(captions)
        self.image_filenames = [f"{CFG.image_prefix}{str(image_filenames[i]).zfill(12)}.jpg" for i in range(len(image_filenames))]
        
        self.encoded_captions = tokenizer(self.captions, padding=True, truncation=True,max_length=CFG.max_length)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self.transforms(image=image)['image']
        # item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        img = np.moveaxis(image, source=-1, destination=0)

        # img = Image.open(f"{CFG.image_path}/{self.image_filenames[idx]}")
        # img = np.asarray(img)
        inputs = self.feature_extractor(img, return_tensors="pt")
        # img = self.transforms(image=img)['image']
        item['image'] = inputs['pixel_values'][0] 
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                # A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                # A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

    