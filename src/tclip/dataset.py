import os
import cv2
import torch
import albumentations as A
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from transformers import ViTFeatureExtractor

def load_data(params):
    train_df, valid_df = prepare_dataframe(params.lang, params.captions_path)
    train_loader = build_loaders(train_df, "train", params)
    valid_loader = build_loaders(valid_df, "valid", params)
    return train_loader, valid_loader

def prepare_dataframe(lang, captions_path):
    # load caption file
    if lang == 'es':
        df = pd.read_csv(f"{captions_path}", encoding = 'utf-8-sig')
    else:
        df = pd.read_csv(f"{captions_path}")

    x = list(set(df['image_id'].values))
    image_ids = np.arange(0, len(x))
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    # valid_ids = image_ids[len(image_ids)-1000:len(image_ids)]
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_images = [x[i] for i in train_ids]
    val_images = [x[i] for i in valid_ids]
    train_df = df[df["image_id"].isin(train_images)].reset_index(drop=True)
    valid_df = df[df["image_id"].isin(val_images)].reset_index(drop=True)
    return train_df, valid_df

def build_loaders(df, mode, params):
    image_ids = df["image_id"].values
    image_filenames = [f"{params.image_path}/{params.image_prefix}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
    # not_avai = []
    # for i in range(len(image_ids)):
    #     if not os.path.isfile(image_filenames[i]):
    #         not_avai.append(i)
    # df = df.drop(df.index[not_avai])
    # image_ids = df["image_id"].values
    # image_filenames = [f"{params.image_path}/{params.image_prefix}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
    # if params.lang == 'it':
    #     dataset = CLIPDataset_resnet(
    #         image_filenames,
    #         df["caption"].values,
    #     )
    # else:
    dataset = CLIPDataset_ViT(
        image_filenames,
        df["caption"].values,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

class CLIPDataset_ViT(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, transforms=None):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.captions = list(captions)
        self.image_filenames = image_filenames
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __getitem__(self, idx):
        image = cv2.imread(self.image_filenames[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(image, source=-1, destination=0)
        inputs = self.feature_extractor(img, return_tensors="pt") # transforms already

        item = {}
        item['image'] = inputs['pixel_values'][0] 
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        return len(self.captions)


class CLIPDataset_resnet(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, transforms=None):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.captions = list(captions)
        self.image_filenames = image_filenames
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        # image = cv2.imread(self.image_filenames[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(self.image_filenames[idx]).convert('RGB')
        # img = np.moveaxis(image, source=-1, destination=0)
        if self.transforms is not None:
            image = self.transforms(image)

        item = {}
        item['image'] = image
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        return len(self.captions)

    