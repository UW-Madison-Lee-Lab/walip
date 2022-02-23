import pdb
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from transformers import ViTFeatureExtractor
from utils.helper import generate_path
import configs


class ImageDataset(Dataset):
    def __init__(self, data, labels, transforms=None, multilabel=False):
        super(ImageDataset, self).__init__()
        self.transforms = transforms
        self.data = data
        self.targets = labels
        self.multilabel = multilabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]
        if self.transforms is not None:
            img = self.transforms(img)
        if not self.multilabel:
            label = int(label)
        return img, label

class ViTDataset(Dataset):
    def __init__(self, dataset, convert_image = False):
        super(ViTDataset, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.data = dataset.data
        self.targets = dataset.targets
        self.convert_image = convert_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.convert_image:
            image = cv2.imread(self.data[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = np.moveaxis(image, source=-1, destination=0)
            img = self.feature_extractor(img, return_tensors="pt")
        else:
            img = self.feature_extractor(self.data[index], return_tensors="pt")
        return img['pixel_values'][0], self.targets[index]


# label is in format 1 3 10 with 11 classes converts to -> [0,1,0,1,0,0,0,0,0,0,1]
def one_hot(label, n_classes):
    l = np.zeros(n_classes)
    indices = [int(x) for x in label.split(" ")]
    l[indices] = 1
    return l

def get_first_label(label):
    indices = [int(x) for x in label.split(" ")]
    return indices[0]


def load_image_dataset(image_name, data_dir='../dataset', preprocess=None, multilabel=False):
    print('.....', '.....', '.....', "Load image dataset ", image_name)
    if image_name.startswith('cifar'):
        if image_name == 'cifar100':
            image_dataset = CIFAR100(data_dir, transform=preprocess, download=True, train=False)
        elif image_name == 'cifar10':
            image_dataset = CIFAR10(data_dir, transform=preprocess, download=True, train=False)
        if preprocess is None:
            image_dataset = ViTDataset(image_dataset)

    elif image_name.startswith("imagenet"):
        if preprocess is None:
            preprocess = transforms.Compose([
                Resize([224], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize(configs.means[image_name], configs.stds[image_name]),
            ])
        image_dataset = ImageFolder(os.path.join(data_dir, f'{image_name}/val/'), preprocess)

    elif image_name == "coco":
        images, labels = load_image_data(image_name, 1000, False, "./", multilabel = multilabel)
        image_dataset = ViTDataset(ImageDataset(images, labels, multilabel=True), convert_image=True)

    return image_dataset



def load_image_data(image_name, num_images, using_filtered_images, src_lang, tgt_lang, preprocess=None, multilabel=False):
    num_classes = configs.num_classes[image_name]
    print('.....', '.....', '.....', "Load image data", image_name, num_images)

    if image_name == 'coco':
        # Expected to have image_id, and labels columns
        captions_path = f"../dataset/coco/captions/en/{configs.caption_names['en']}"
        df = pd.read_csv(f"{captions_path}")
        image_ids = df["image_id"].values
        if multilabel:
            labels = df["labels"].apply(lambda x: one_hot(x, configs.num_classes[image_name])).values
        else:
            labels = df["labels"].apply(get_first_label).values
        
        indices = np.arange(len(image_ids))
        np.random.seed(42)
        np.random.shuffle(indices)
        image_ids = image_ids[indices]
        final_labels = labels[indices]

        image_path = f"../dataset/coco/images/{configs.image_folders['en']}"
        image_filenames = [f"{image_path}/{configs.image_prefixes['en']}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
        images = image_filenames
        return images[:num_images], final_labels[:num_images]


    ####------------ other datasets ------------ ########
    image_dataset = load_image_dataset(image_name, preprocess=preprocess)
    if using_filtered_images:
        fpath = generate_path('img_shared_index', {'image_data': image_name, 'src_lang': src_lang, 'tgt_lang': tgt_lang})
        dct = np.load(fpath, allow_pickle=True).item()
        images = []
        for idx in list(dct.values()):
            images += [image_dataset[idx[i]][0].numpy() for i in range(min(num_images, len(idx)))]
        return np.stack(images, axis=0)
        
    if image_name in ['cifar100', 'cifar10']:
        labels = np.asarray(image_dataset.targets)
    elif image_name.startswith('imagenet'):
        label_path = generate_path('img_label', {'image_data': image_name})
        if os.path.isfile(label_path):
            labels = np.load(label_path, allow_pickle=True)
        else:
            dataloader = DataLoader(image_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=4)
            labels = []
            for batch in tqdm(dataloader):
                labels.append(batch[1].numpy())
            labels = np.stack(labels, axis=0)
            np.save(label_path, np.asarray(labels))
        labels = labels.flatten()
    # final images
    images = []
    for c in range(num_classes):
        indices = np.argwhere(labels == c)
        # np.random.shuffle(indices)
        images += [image_dataset[indices[i][0]][0] for i in range(num_images)]
    images = np.stack(images, axis=0)
    return images