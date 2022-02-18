model_names = {
    "en": "bert-base-uncased",
	"en2": "bert-base-uncased",
    "es": "../pretrained/uncased/",
    "it": "dbmdz/bert-base-italian-uncased",
	'sw': 'something'
}

image_folders = {
    "en": "train2014",
	"en2": '',
    "es": "spanish_images",
    "it": "train2014",
	"sw": ''
}

image_prefixes = {
    'en': 'COCO_train2014_',
	'en2': '',
    'es': '',
    'it': 'COCO_train2014_',
	'sw': ''
}
caption_names = {
    "en": "processed_captions_train2014.csv",
    "es": "es/",
    "it": "processed_captions_train2014.csv",
	"sw": ''
}


means = {
	'cifar10': (0.4914, 0.4822, 0.4465),
	'cifar100': (0.5071, 0.4867, 0.4408),
	'imagenet': (0.48145466, 0.4578275, 0.40821073),
	'tiny': (0.48145466, 0.4578275, 0.40821073)

}

stds = {
	'cifar10': (0.2023, 0.1994, 0.2010),
	'cifar100': (0.2675, 0.2565, 0.2761),
	'imagenet': (0.26862954, 0.26130258, 0.27577711),
	'tiny': (0.26862954, 0.26130258, 0.27577711)
}



num_classes = {
	'cifar10': 10,
	'cifar100': 100,
	'tiny': 200,
	'imagenet': 1000,
	'imagenet-sketch': 999,
	'imagenet-r': 200,
	'coco': 80
}

delimiters = {
	'cifar10': ' ',
	'cifar100': ' ',
	'noun': ' ',
	'imagenet': ' | ',
	'imagenet-r': ' | ',
	'composite': ' | ',
	'wiki': ' ', 
	'coco': ' '
}

image_batchsizes = {'imagenet':4, 'cifar100': 128, 'cifar10':128}


#### constants
FINGERPRINT = 'fp'
FASTTEXT = 'fasttext'
GANFASTTEXT = 'ganfasttext200'
CLIPTEXT = 'cliptext'