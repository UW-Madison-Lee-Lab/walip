# from matplotlib import pyplot as plt
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
    print('Accuracy: {:.4f}'.format(acc))
    return acc, wrong_pairs


def log(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)