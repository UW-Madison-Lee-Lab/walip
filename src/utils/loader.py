
import os, sys
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from transformers import ViTFeatureExtractor
import pandas as pd
import cv2
import configs


def load_vocabs_from_pairs(opts):
    fpath = opts.txt_dir + f'{opts.word_data}_{opts.src_lang}_{opts.tgt_lang}_{opts.data_mode}.txt'
    vocabs = {'src':[], 'tgt': []}
    with open(fpath) as f:
        lines = f.readlines()
    for l in lines:
        if configs.delimiters[opts.word_data] is not None:
            x, y = l.strip().lower().split()
        else:
            x, y = l.strip().lower().split()
        vocabs['src'].append(x)
        vocabs['tgt'].append(y)
    return vocabs

def load_vocabs(opts, lang):
    fpath = opts.txt_dir + f'{opts.word_data}_{lang}_{opts.data_mode}.txt'
    if not os.path.isfile(fpath):
        print("------> Error: Load vocabs", fpath, "file doesn't exist!!!")
        sys.exit('Done')
       
    with open(fpath) as f:
        lines = f.readlines()
    vocabs = [] # order
    for desc in lines:
        desc = desc.strip().lower()
        vocabs.append(desc)
    return vocabs

def combine_files(opts):
    vocabs_src = load_vocabs(opts, opts.src_lang)
    vocabs_tgt = load_vocabs(opts, opts.tgt_lang)
    fpath = opts.txt_dir + f'{opts.word_data}_{opts.src_lang}_{opts.tgt_lang}_{opts.data_mode}.txt'
    f= open(fpath, "w") 
    for i in range(len(vocabs_src)):
        f.write(f"{vocabs_src[i]}{configs.delimiters[opts.word_data]}{vocabs_tgt[i]}\n")
    f.close()



def get_word2id(vocab):
    word2id = {}
    for i in range(len(vocab)):
        word2id[vocab[i]] = i
    return word2id

def load_image_dataset(image_name, data_dir='../dataset'):
    print('.....', '.....', '.....', "Load image dataset ", image_name)
    preprocess = transforms.Compose([
        Resize([224], interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(configs.means[image_name], configs.stds[image_name]),
    ])

    if image_name.startswith('cifar'):
        preprocess = None
        if image_name == 'cifar100':
            from torchvision.datasets import CIFAR100
            image_dataset = CIFAR100(data_dir, transform=preprocess, download=True, train=False)
        elif image_name == 'cifar10':
            from torchvision.datasets import CIFAR10
            image_dataset = CIFAR10(data_dir, transform=preprocess, download=True, train=False)

        image_dataset = ViTDataset(image_dataset)

    elif image_name in ['tiny', 'imagenet']:
        import h5py as h5
        data_dir = '~/scratch/tprojects/repgan/data/'
        hdf5_path = data_dir + f'{image_name}/{image_name}_valid.h5'
        if os.path.isfile(hdf5_path):
            print('Loading %s into memory...' % hdf5_path)
            with h5.File(hdf5_path, 'r') as f:
                imgs = np.asarray(f.get('imgs'))
                labels = np.asarray(f.get('labels'))
            image_dataset = ImageDataset(imgs, labels, preprocess)
        else:
            from torchvision.datasets import ImageFolder
            image_dataset = ImageFolder(data_dir + 'imagenet/test/', preprocess)
            # imgs = []
            # for i in tqdm(chunks(1, range(len(image_data)))):
            #     imgs.append(image_data[i][0])
            # labels = [image_data[i][1] for i in range(len(image_data))]
            # from IPython import embed; embed()
            # hf = h5.File(hdf5_path, 'w')
            # hf.create_dataset('imgs', data=np.stack(imgs, axis=0))
            # hf.create_dataset('labels', data=np.asarray(labels))
            # hf.close()

    return image_dataset



class ImageDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        super(ImageDataset, self).__init__()
        self.transforms = transforms
        self.data = data
        self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]
        if self.transforms is not None:
            img = self.transforms(img)
        label = int(label)
        return img, label

class ViTDataset(Dataset):
    def __init__(self, dataset):
        super(ViTDataset, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.data = dataset.data
        self.targets = dataset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.feature_extractor(self.data[index], return_tensors="pt")
        return img['pixel_values'][0], self.targets[index]

def load_image_data(image_name, num_images, using_filtered_images, img_dir):
    num_classes = configs.num_classes[image_name]
    print('.....', '.....', '.....', "Load image data", image_name, num_images)
    if not image_name == 'coco':
        image_dataset = load_image_dataset(image_name)
    print('Load', num_images, 'images from dataset')
    if using_filtered_images:
        fpath = os.path.join(img_dir, f'{image_name}_en_it_index.npy')
        if os.path.isfile(fpath):
            dct = np.load(fpath, allow_pickle=True).item()
            indices = list(dct.values())
            images = [image_dataset[idx][0].numpy() for idx in indices]
        else:
            with open(img_dir + image_name + '_index.txt') as f:
                lines = f.readlines()
            d = {}
            for l in lines:
                k, v = l.strip().split(' ')
                k, v = int(k), int(v)
                if k in d:
                    d[k].append(v)
                else:
                    d[k] = [v]
            images = []
            for c in range(num_classes):
                for i in range(num_images):
                    idx = d[c][i]
                    images.append(image_dataset[idx][0])
    else:
        images = []
        # pick one image per class
        if image_name in ['cifar100', 'cifar10']:
            for c in range(num_classes):
                indices = np.argwhere(np.asarray(image_dataset.targets) == c)
                images += [image_dataset[indices[i][0]][0] for i in range(num_images)]
        elif image_name.startswith('imagenet'):
            label_path = img_dir + f'label_{image_name}.npy'
            if os.path.isfile(label_path):
                labels = np.load(label_path, allow_pickle=True)
            else:
                dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
                labels = []
                for batch in tqdm(dataloader):
                    labels.append(batch[1].numpy())
                labels = np.stack(labels, axis=0)
                np.save(label_path, np.asarray(labels))
            feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            for c in range(num_classes):
                indices = np.argwhere(labels == c)
                img = image_dataset[indices[0][0]][0]#.numpy()
                inputs = feature_extractor(img, return_tensors="pt") 
                # images += [ for i in range(num_images)]
                images.append(inputs['pixel_values'][0])
        elif image_name == 'coco':
            captions_path = f"../dataset/coco/captions/en/{configs.caption_names['en']}"
            df = pd.read_csv(f"{captions_path}")
            image_ids = df["image_id"].values
            image_path = f"../dataset/coco/images/{configs.image_folders['en']}"
            image_filenames = [f"{image_path}/COCO_train2014_{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
            feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            images = []
            indices = np.arange(len(image_ids))
            np.random.shuffle(indices)
            random_indices = indices[:100] 
            for k in random_indices:
                image = cv2.imread(image_filenames[k])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = np.moveaxis(image, source=-1, destination=0)
                inputs = feature_extractor(img, return_tensors="pt") # transforms already
                images.append(inputs['pixel_values'][0]) 
            
        images = np.stack(images, axis=0)
    return images