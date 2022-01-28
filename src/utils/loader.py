from torchvision import transforms
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
import os, sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from funcy import chunks
from tqdm import tqdm
from PIL import Image
import configs


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



def load_vocabs_from_pairs(data_name, langs, mode):
    fpath = configs.paths['txt_dir'] + f'{data_name}_{langs[0]}_{langs[1]}_{mode}.txt'
    vocabs = {langs[0]:[], langs[1]: []}
    with open(fpath) as f:
        lines = f.readlines()

    for l in lines:
        if configs.delimiters[data_name] is not None:
            x, y = l.strip().lower().split(separator=configs.delimiters[data_name])
        else:
            x, y = l.strip().lower().split()
        vocabs[langs[0]].append(x)
        vocabs[langs[1]].append(y)
    return vocabs

def load_vocabs(data_name, lang, mode):
    fpath = configs.paths['txt_dir']+ '{}_{}_{}.txt'.format(data_name, lang, mode)
    print(fpath)
    if not os.path.isfile(fpath):
        sys.exit("File doesn't exist!!!")
       
    with open(fpath) as f:
        lines = f.readlines()
    vocabs = [] # order
    for desc in lines:
        desc = desc.strip().lower()
        vocabs.append(desc)
    return vocabs

def combine_files(data_name, langs, mode):
    vocabs = {}
    for i in range(2):
        vocabs[i] = load_vocabs(data_name, langs[i], mode)
    
    fpath = configs.paths['txt_dir'] + f'{data_name}_{langs[0]}_{langs[1]}_{mode}.txt'
    f= open(fpath, "w") 
    for i in range(len(vocabs[0])):
        f.write(f"{vocabs[0][i]}{configs.delimiters[data_name]}{vocabs[1][i]}\n")
    f.close()
    print('Done combining')



def get_word2id(vocab):
    word2id = {}
    for i in range(len(vocab)):
        word2id[vocab[i]] = i
    return word2id

def load_image_dataset(image_name):
    print("Load image dataset ", image_name)
    preprocess = transforms.Compose([
        Resize([224], interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(configs.means[image_name], configs.stds[image_name]),
    ])
    if image_name == 'cifar100':
        from torchvision.datasets import CIFAR100
        image_dataset = CIFAR100('../data', transform=preprocess, download=True, train=False)
    elif image_name == 'cifar10':
        from torchvision.datasets import CIFAR10
        image_dataset = CIFAR10('../data', transform=preprocess, download=True, train=False)
    elif image_name in ['tiny', 'imagenet']:
        import h5py as h5
        hdf5_path = f'../../repgan/data/{image_name}/{image_name}_valid.h5'
        if os.path.isfile(hdf5_path):
            print('Loading %s into memory...' % hdf5_path)
            with h5.File(hdf5_path, 'r') as f:
                imgs = np.asarray(f.get('imgs'))
                labels = np.asarray(f.get('labels'))
            image_dataset = ImageDataset(imgs, labels, preprocess)
        else:
            from torchvision.datasets import ImageFolder
            data_dir = '/mnt/nfs/work1/mccallum/dthai/tuan_data/' + f'{image_name}/val'
            image_dataset = ImageFolder(data_dir, preprocess)
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

def load_image_data(image_name):
    num_images = configs.num_images
    num_classes = configs.num_classes[image_name]
    print("Load image data ", image_name, num_images)
    image_dataset = load_image_dataset(image_name)
    print('Load', num_images, 'images from dataset')
    if configs.flags["using_filtered_images"]:
        fpath = configs.paths['img_dir']  + '{}_{}_{}_index.npy'.format(image_name, configs.langs['src'], configs.langs['tgt'])
        if os.path.isfile(fpath):
            dct = np.load(fpath, allow_pickle=True).item()
            indices = list(dct.values())
            images = [image_dataset[idx][0].numpy() for idx in indices]
        else:
            with open(configs.paths['img_dir'] + image_name + '_index.txt') as f:
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
        else:
            label_path = configs.paths['img_dir'] + f'label_{image_name}.npy'
            if os.path.isfile(label_path):
                labels = np.load(label_path, allow_pickle=True)
            else:
                dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
                labels = []
                for batch in tqdm(dataloader):
                    labels.append(batch[1].numpy())
                labels = np.stack(labels, axis=0)
                np.save(label_path, np.asarray(labels))
            for c in range(num_classes):
                indices = np.argwhere(labels == c)
                images += [image_dataset[indices[i][0]][0].numpy() for i in range(num_images)]
        images = np.stack(images, axis=0)
    return images