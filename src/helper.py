import pdb


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



def load_vocabs(data_name, lang='en'):
	with open('../dicts/texts/{}_{}.txt'.format(data_name, lang)) as f:
		lines = f.readlines()
	vocabs = [] # order
	for desc in lines:
		desc = desc.strip().lower()
		vocabs.append(desc)
	return vocabs

def load_data_from_one_files(data_name, langs=['en', 'it'], mode='test'):
    fpath = '../dicts/texts/{}_{}_{}_{}.txt'.format(data_name, langs[0], langs[1], mode)
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


def load_data_from_two_files(data_name, langs=['en', 'it']):
    vocabs = [[], []]
    translation = {}
    lines = []
    for l in langs:
        fpath = '../dicts/texts/{}_{}.txt'.format(data_name, l)
        print('loading data from' + fpath)
        with open(fpath) as f:
            lines.append(f.readlines())

    for (x, y) in zip(lines[0], lines[1]):
        x = x.strip().lower()
        y = y.strip().lower()
        vocabs[0].append(x)
        vocabs[1].append(y)
        translation[x] = [y]
    return vocabs, translation
