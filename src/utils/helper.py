# from matplotlib import pyplot as plt
import os
from itertools import chain
from torchvision.utils import save_image

def plotW(score, name):
    plt.imshow(score)
    plt.colorbar()
    plt.savefig('../results/W_{}.png'.format(name))
    plt.close()

def save_images(samples, save_path, nrows=0):
    if nrows == 0:
        bs = samples.shape[0]
        nrows = int(bs**.5)
    save_image(samples.cpu(), save_path, nrow=nrows, normalize=True)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_accuracy(dico, col_ind):
    s = 0
    wrong_pairs = []
    for i in range(dico.shape[0]):
        if col_ind[i] == dico[i, 1]:
            s+=1
        else:
            wrong_pairs.append([i, col_ind[i], dico[i, 1]])
            # print(i, '->', col_ind[i], vocabs[0][i], '->', vocabs[0][col_ind[i]])
    acc = s/dico.shape[0] * 100
    print('Accuracy: {:.4f}/100'.format(acc))
    return acc, wrong_pairs


def log(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    kwargs = flatten_dict(kwargs)
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(train_configs, model_configs):
    cfgs = {}
    for k, v in chain(train_configs.items(), model_configs.items()):
        cfgs[k] = v

    class cfg_container: pass
    cfg_container.train_configs = train_configs
    cfg_container.model_configs = model_configs
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container


def try_make_dir(d):
    if not os.path.isdir(d):
        # os.mkdir(d)
        os.makedirs(d) # nested is allowed

def get_basename(fpath):
    basename = os.path.basename(fpath)
    return basename.split('.')[0]

def generate_path(ftype, opts):
    root = '../dicts/'
    if 'selected' in opts:
        s = 's' if opts["selected"] else 'u'
    if ftype.startswith('emb'):
        root += 'embeddings/'
        if ftype == 'emb_txt':
            fdir = f'txt_emb/{opts["word_data"]}/'
            fname = f'txt_emb_{opts["word_data"]}_{opts["src_lang"]}_{opts["tgt_lang"]}_{opts["lang"]}_{opts["data_mode"]}.npy'
        elif ftype == 'emb_img':
            fdir = f'img_emb/{opts["image_data"]}/'
            fname = f'img_emb_{opts["image_data"]}_{opts["lang"]}_k{opts["num_images"]}_{s}.npy'
        elif ftype == 'emb_fp':
            prefix = f'{opts["image_data"]}_{s}_{opts["word_data"]}'
            fdir = f'fp/{prefix}/'
            fname = f'fp_{prefix}_{opts["src_lang"]}_{opts["tgt_lang"]}_{opts["lang"]}_{opts["data_mode"]}.npy'
        elif ftype == 'emb_fasttext': # fasttext
            fdir = f'fasttext/{opts["word_data"]}/'
            fname = f'fasttext_{opts["word_data"]}_{opts["src_lang"]}_{opts["tgt_lang"]}_{opts["lang"]}_{opts["data_mode"]}.npy'
    elif ftype.startswith('img'):
        fdir = f'images/{opts["image_data"]}/'
        if ftype == 'img':
            fname = f'img_{opts["image_data"]}_{opts["lang"]}_k{opts["num_images"]}_{s}.npy'
        elif ftype == 'img_label':
            fname = f'label_{opts["image_data"]}.npy'
        elif ftype == 'img_index':
            fname = f'index_{opts["image_data"]}_{opts["lang"]}.npy'
        elif ftype == 'img_shared_index':
            fname = f'shared_index_{opts["image_data"]}_{opts["src_lang"]}_{opts["tgt_lang"]}.npy'
    elif ftype == 'txt_single':
        fdir = f'texts/{opts["word_data"]}/'
        fname = f'{opts["word_data"]}_{opts["src_lang"]}_{opts["tgt_lang"]}_{opts["lang"]}_{opts["data_mode"]}.txt'
    elif ftype == 'txt_pair':
        fdir = f'texts/{opts["word_data"]}/'
        fname = f'{opts["word_data"]}_{opts["src_lang"]}_{opts["tgt_lang"]}_{opts["data_mode"]}.txt'
    
    
    folder = os.path.join(root, fdir)
    try_make_dir(folder)
    return os.path.join(folder, fname)