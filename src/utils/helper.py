
from matplotlib import pyplot as plt

def plotW(score, name):
    plt.imshow(score)
    plt.colorbar()
    plt.savefig('../results/W_{}.png'.format(name))
    plt.close()




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


def get_accuracy(vocabs, translation, col_ind):
    assert len(vocabs) == 2
    s = 0
    for i in range(len(vocabs[0])):
        if vocabs[1][col_ind[i]] in translation[vocabs[0][i]]:
            s+=1
        else:
            print(i, ':', col_ind[i], vocabs[0][i], vocabs[1][col_ind[i]])
    acc = s/len(vocabs[0]) * 100
    print('Accuracy: {:.4f}'.format(acc))
    return acc