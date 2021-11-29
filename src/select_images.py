
import os
import numpy as np
from funcy import chunks
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

import clip
from templates import generate_texts, templates
from mclip import multilingual_clip
from ops import get_clip_image_features, get_clip_text_features, cal_probs_from_features
from helper import load_vocabs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

means = {
	'cifar10': (0.4914, 0.4822, 0.4465),
	'cifar100': (0.5071, 0.4867, 0.4408),
	'imagenet': (0.48145466, 0.4578275, 0.40821073)
}

stds = {
	'cifar10': (0.2023, 0.1994, 0.2010),
	'cifar100': (0.2675, 0.2565, 0.2761),
	'imagenet': (0.26862954, 0.26130258, 0.27577711)
}


clip_name = "ViT-B/32"
clip_model, preprocess = clip.load(clip_name)
clip_model.cuda().eval()

def get_embedding(data_name, lang='eng'):

	clip_path = '../dicts/{}_clip_embedding_{}.npy'.format(data_name, lang)
	if os.path.isfile(clip_path):
		clip_based_text_features = np.load(clip_path, allow_pickle=True)
	else:
		vocabs = load_vocabs(data_name, lang)
		texts = generate_texts(templates[lang], vocabs)
		# embeds
		if lang == 'eng':
			text_model = clip_model
		else:
			text_model = multilingual_clip.load_model('M-BERT-Base-ViT-B')
			text_model.eval()
		
		clip_embeds = []
		bz = 50 * len(templates[lang])
		for t in tqdm(chunks(bz, texts)):
			text_features = get_clip_text_features(t, text_model, lang == 'eng', len(templates[lang]))
			clip_embeds.append(text_features.cpu().numpy())
		# save text_features -- 
		clip_based_text_features = np.concatenate(clip_embeds, axis=0)
		np.save(clip_path, clip_based_text_features)

	# get fingerprints
	image_features = get_clip_image_features(clip_name, 'imagenet', 1, clip_model)
	text_features = torch.Tensor(clip_based_text_features).cuda()
	# CLIP Temperature scaler
	logit_scale = clip_model.logit_scale.exp().float()
	fingerprints = cal_probs_from_features(image_features, text_features, logit_scale)
	np.save('../dicts/{}_{}_embedding_{}'.format(data_name, 'imagenet', lang), fingerprints)

def filter_images(data_name, lang='eng'):

	bs = {'imagenet':32, 'cifar100': 128, 'cifar10':128}[data_name]
	nclasses = {'imagenet': 1000, 'cifar100': 100, 'cifar10':10}[data_name]

	vocabs = load_vocabs(data_name, lang)	
	texts = generate_texts(templates[lang], vocabs)
	if lang == 'eng':
		text_features = get_clip_text_features(texts, clip_model, True, len(templates[lang]))
	else:
		# text_model = multilingual_clip.load_model('M-BERT-Distil-40')
		text_model = multilingual_clip.load_model('M-BERT-Base-ViT-B')
		text_model.eval()
		text_features = get_clip_text_features(texts, text_model, False, len(templates[lang]))
	text_features = text_features.cuda()

	val_preprocess = transforms.Compose([
			Resize([224], interpolation=InterpolationMode.BICUBIC),
			CenterCrop(224),
			ToTensor(),
			Normalize(means[data_name], stds[data_name]),
		])

	if data_name == 'cifar100':
		dataset = CIFAR100('../data', transform=val_preprocess, download=True, train=False)
	elif data_name == 'imagenet':
		data_dir = '/mnt/nfs/work1/mccallum/dthai/tuan_data/imagenet/val'
		dataset = ImageFolder(data_dir, val_preprocess)
	dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=4)

	indices = {}
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		targets = Variable(labels).long().cuda()
		images = Variable(inputs).cuda()
		with torch.no_grad():
			image_features = clip_model.encode_image(images).float()
			image_features /= image_features.norm(dim=-1, keepdim=True)

		logits = image_features @ text_features.t()
		_, pred = logits.topk(1, 1, True, True)
		pred = pred.t()
		correct = pred.eq(targets.view(1, -1).expand_as(pred))
		correct = correct[0]
		
		inds = np.where(correct.cpu().numpy() == True)[0]
		for x in inds:
			t = labels[x].item()
			if t not in indices:
				indices[t] = []
			indices[t].append(x + batch_idx * bs)

	fname = '../dicts/{}_{}_correct_index'.format(data_name, lang)
	np.save(fname, indices)
	f = open(fname + ".txt", 'w')
	for i in range(nclasses):
		if i in indices:
			f.write("{} {}\n".format(i, indices[i]))
		else:
			print(i, 'not in indices')
	f.close()


def find_interesection(data_name, langs=['eng', 'ita']):
	def intersect(lst1, lst2):
		lst3 = [value for value in lst1 if value in lst2]
		return lst3

	indices = []
	for l in langs:
		fpath = '../dicts/{}_{}_correct_index.npy'.format(data_name, l)
		indices.append(np.load(fpath, allow_pickle=True).item())

	keys = intersect(list(indices[0].keys()), list(indices[1].keys()))
	ans = {}
	for k in keys:
		values = intersect(indices[0][k], indices[1][k])
		if len(values) > 0:
			ans[k] = values[0]
		else:
			print(k) # 36
	
	np.save('../dicts/image_{}_{}_{}_index'.format(data_name, langs[0], langs[1]), ans)


if __name__ == '__main__':
	# filter_images('cifar100', 'eng')
	# find_interesection('cifar100')
	get_embedding('wiki_200k', 'eng')