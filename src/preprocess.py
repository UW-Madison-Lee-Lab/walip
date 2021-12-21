from utils.filter_images import find_correct_images, find_interesection
import configs

data = 'imagenet'

dict_pth = '../dicts/'
configs.paths['emb_dir'] = dict_pth + 'embeddings/'
configs.paths['img_dir'] = dict_pth + 'images/{}/'.format(data)
configs.paths['txt_dir'] = dict_pth + 'texts/{}/'.format(data)

find_correct_images('en', data)
find_correct_images('it', data)
find_interesection(data)