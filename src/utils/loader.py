from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
import os
import numpy as np
import configs

    
def load_vocabs(data_name, lang, mode):
    fpath = configs.text_dir_path + '{}/{}_{}_{}.txt'.format(data_name, data_name, lang, mode)
    with open(fpath) as f:
        lines = f.readlines()
    vocabs = [] # order
    for desc in lines:
        desc = desc.strip().lower()
        vocabs.append(desc)
    return vocabs

def load_data_from_two_files(data_name, langs, mode):
    vocabs = [[], []]
    translation = {}
    lines = []
    for lang in langs:
        fpath = configs.text_dir_path + '{}/{}_{}_{}.txt'.format(data_name, data_name, lang, mode)
        with open(fpath) as f:
            lines.append(f.readlines())

    for (x, y) in zip(lines[0], lines[1]):
        x = x.strip().lower()
        y = y.strip().lower()
        vocabs[0].append(x)
        vocabs[1].append(y)
        translation[x] = [y]
    return vocabs, translation

def combine_files(data_name, langs, mode):
    vocabs, _ = load_data_from_two_files(data_name, langs, mode)
    fpath = configs.text_dir_path + '{}/{}_{}_{}_{}.txt'.format(data_name, data_name, langs[0], langs[1], mode)
    f= open(fpath, "w") 
    for i in range(len(vocabs[0])):
        f.write("{} {}\n".format(vocabs[0][i], vocabs[1][i]))
    f.close()
    print('Done combining')

def load_data_from_one_file(data_name, langs, mode):
    fpath = configs.text_dir_path + '{}/{}_{}_{}_{}.txt'.format(data_name, data_name, langs[0], langs[1], mode)
    if not os.path.isfile(fpath):
        combine_files(data_name, langs, mode)
    vocabs = [[], []]
    translation = {}
    with open(fpath) as f:
        lines = f.readlines()

    for l in lines:
        x, y = l.strip().lower().split(' ')
        vocabs[0].append(x)
        vocabs[1].append(y)
        if x in translation:
            # pass
            translation[x].append(y)
        else:
            translation[x] = [y]
    # for i in range(2):
    #     vocabs[i] = list(vocabs[i])
    return vocabs, translation

def load_vocab_translation(txt_data_name, langs, mode):
    vocabs, translation = load_data_from_one_file(txt_data_name, langs, mode)
    return vocabs, translation

def get_word2id(vocabs):
    assert len(vocabs) == 2
    word2ids = {0:{}, 1:{}}
    for k in range(2):
        for i in range(len(vocabs[k])):
            word2ids[k][vocabs[k][i]] = i
    return word2ids

def load_image_dataset(image_name):
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
    elif image_name == 'tiny':
        import h5py as h5
        from PIL import Image
        hdf5_path = '../../repgan/data/tiny/tiny_valid.h5'
        print('Loading %s into memory...' % hdf5_path)
        with h5.File(hdf5_path, 'r') as f:
            imgs = np.asarray(f.get('imgs'))
            labels = np.asarray(f.get('labels'))
    else:
        from torchvision.datasets import ImageFolder
        data_dir = '../../repgan/data/imagenet/val'
        image_dataset = ImageFolder(data_dir, preprocess)
    return image_dataset

def load_image_data(image_name):
    num_images = configs.num_images
    num_classes = configs.num_classes[image_name]
    image_dataset = load_image_dataset(image_name)

    if configs.flags["using_filtered_images"]:
        fpath = '../dicts/images/{}_{}_{}_index.npy'.format(image_name, configs.langs['src'], configs.langs['tgt'])
        if os.path.isfile(fpath):
            dct = np.load(fpath, allow_pickle=True).item()
            indices = list(dct.values())
            images = [image_dataset[idx][0] for idx in indices]
        else:
            with open('../dicts/images/{}_index.txt'.format(image_name)) as f:
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
                for i in range(num_images):
                    images.append(image_dataset[indices[i][0]][0])
        else:
            for c in range(num_classes):
                indices = np.argwhere(labels == c)
                for i in range(num_images):
                    image = imgs[indices[i][0]]
                    image = Image.fromarray(image, 'RGB')
                    image = preprocess(image)
                    images.append(image)
    images = np.stack(images, axis=0)
    return images