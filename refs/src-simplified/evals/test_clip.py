import numpy as np
import clip
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable




def evaluate():
	clip_name = "ViT-B/32"
	clip_model, preprocess = clip.load(clip_name)
	clip_model.cuda().eval()
	image_dataset = CIFAR10('../data', transform=preprocess, download=True, train=False)
	dataloader = DataLoader(image_dataset, batch_size=512, shuffle=False, drop_last=True, num_workers=4)


	# texts = [f"This is a photo of a " + desc for desc in image_dataset.classes]
	with open('../cifar10_en.txt') as f:
		lines = f.readlines()
	texts = []
	for desc in lines:
		desc = desc.strip().lower()
		t = [templates[i].format(desc) for i in range(len(templates))]
		texts += t

	text_tokens = clip.tokenize(texts).cuda()
	with torch.no_grad():
		text_features = clip_model.encode_text(text_tokens).float()
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)
		# ensemble : (B xk)xd)
		text_features = text_features.view(len(lines), len(templates), text_features.shape[-1])
		text_features = text_features.mean(dim=1)

	top5, top1 = AverageMeter(), AverageMeter()
	indices = {}
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		# print(batch_idx)
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
				indices[t] = x + batch_idx * 128

		# if len(indices) == 100:
		# 	print('Done')
		# 	t = [indices[i] for i in range(100)]
		# 	np.saves('../cifar10_index.npy', np.asarray(t))
		# 	from IPython import embed; embed()
		# 	break
				
		precs = accuracy(logits, targets, topk=(1, 5))
		top1.update(precs[0].item(), inputs.size(0))
		top5.update(precs[1].item(), inputs.size(0))

	print(top1.avg, top5.avg)
	f = open("../cifar100_index.txt", 'w')
	for k, v in indices.items():
		f.write("{} {}\n".format(k, v))
	f.close()



if __name__ == '__main__':
	evaluate()



