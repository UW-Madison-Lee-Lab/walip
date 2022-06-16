import numpy as np
import torch, os
import torch.nn.functional as F
from funcy import chunks
from tqdm import tqdm
from utils.image_loader import load_image_data
from utils.helper import save_images, generate_path, get_basename
from models.templates import prompts, generate_texts
from models.ops import load_models
import configs


class ClipEmbedding():
    def __init__(self, emb_type, lang, data_mode, opts):
        self.emb_type, self.lang, self.data_mode, self.opts = emb_type, lang, data_mode, opts
        print(self.lang, emb_type, 'embedding')
        self.emb_path = generate_path('emb_' + emb_type, {'lang': lang, 'src_lang': opts.src_lang, 'tgt_lang': opts.tgt_lang, 'word_data': opts.word_data, 'image_data': opts.image_data, 'data_mode': data_mode, 'selected': opts.using_filtered_images, 'num_images': opts.num_images})
        self.model = None
    
    def load_clip_model(self):
        model_name = configs.model_names[self.lang]
        self.model, self.logit_scale, self.preprocess = load_models(self.lang, model_name, 'coco', device=self.opts.device, large_model=self.opts.large_model)
    
    def set_logit_scale(self, value):
        self.logit_scale = value

    def load_embedding(self, vocabs=None):
        if self.opts.reuse_embedding:
            if os.path.isfile(self.emb_path):
                print('.....', 'Reuse emb', get_basename(self.emb_path))
                return np.load(self.emb_path, allow_pickle=True)
            else:
                print("No embedding exists!!!")
        ########### New Embeddings ############
        if self.emb_type in [configs.FASTTEXT, configs.GLOBE, configs.HOWTEXT]:
            embs = np.load(self.emb_path, allow_pickle=True)
        else:
            print('.....', "New embedding", get_basename(self.emb_path))
            txt_embs = self.load_clip_txt_emb(vocabs)
            if self.emb_type == configs.FINGERPRINT:
                img_embs = self.load_clip_img_emb()
                embs = self.load_fingerprint(img_embs, torch.from_numpy(txt_embs).to(self.opts.device))
            else:
                embs = txt_embs
            np.save(self.emb_path, embs) 
        return embs

    def load_clip_img_emb(self):
        img_emb_pth = generate_path('emb_img', {'lang': self.lang, 'image_data': self.opts.image_data, 'selected': self.opts.using_filtered_images, 'num_images': self.opts.num_images})
        if self.opts.reuse_image_embedding and os.path.isfile(img_emb_pth):
            print('.....', '.....', "Reuse img emb", get_basename(img_emb_pth))
            img_embs = np.load(img_emb_pth, allow_pickle=True)
            img_embs = torch.Tensor(img_embs).to(self.opts.device)
        else:
            if self.model is None:
                self.load_clip_model()
            print('.....', '.....', "New image embedding") 
            ########### New Embeddings ############
            img_pth = generate_path('img', {'lang': self.lang, 'image_data': self.opts.image_data, 'selected': self.opts.using_filtered_images, 'num_images': self.opts.num_images})
            if self.opts.reuse_image_data and os.path.isfile(img_pth): 
                print('.....', '.....', '.....', "Reuse img data") 
                images = np.load(img_pth, allow_pickle=True)
            else:
                print('.....', '.....', '.....', "New img data") 
                images = load_image_data(self.opts.image_data, self.opts.num_images, self.opts.using_filtered_images, self.opts.src_lang, self.opts.tgt_lang, self.preprocess)
                np.save(img_pth, images)
            images = torch.Tensor(images)
            # save image 
            # save_images(images, f'../results/base_images_{self.str_filter}.png', nrows=int(images.shape[0]**0.5))
            ### Model embeddings
            list_img_embs = []
            with torch.no_grad():
                for batch_ids in tqdm(chunks(32, range(images.shape[0]))):
                    img_embs = self.model.encode_image(images[batch_ids, ...].to(self.opts.device))
                    img_embs = F.normalize(img_embs, dim=-1)
                    list_img_embs.append(img_embs)    
            img_embs = torch.cat(list_img_embs, dim=0)
            # save embs
            np.save(img_emb_pth, img_embs.cpu().numpy())
        return img_embs

    def load_clip_txt_emb(self, vocabs=None):
        txt_emb_pth = generate_path('emb_txt', {'lang': self.lang, 'src_lang': self.opts.src_lang, 'tgt_lang': self.opts.tgt_lang, 'word_data': self.opts.word_data, 'data_mode': self.data_mode})
        if self.opts.reuse_text_embedding and os.path.isfile(txt_emb_pth):
            print('.....', '.....', "Reuse txt emb", get_basename(txt_emb_pth))
            txt_embs = np.load(txt_emb_pth, allow_pickle=True)
        else:
            if self.model is None:
                self.load_clip_model()
            print('.....', '.....',  "New txt emb") 
            texts = generate_texts(prompts[self.opts.image_data][self.lang], vocabs, k=self.opts.num_prompts)
            txt_embs = []
            K = len(prompts[self.opts.image_data][self.lang]) if -1 == self.opts.num_prompts else self.opts.num_prompts
            bs = 128  * K
            for batch_texts in tqdm(chunks(bs, texts)):
                with torch.no_grad():
                    batch_txt_embs = self.model.encode_text(batch_texts)
                    # ensemble
                    batch_size = len(batch_texts) // K
                    batch_txt_embs = batch_txt_embs.view(batch_size, K, batch_txt_embs.shape[-1])
                    batch_txt_embs = batch_txt_embs.mean(dim=1)
                    # normalize after averaging
                    batch_txt_embs = F.normalize(batch_txt_embs, dim=-1)
                txt_embs.append(batch_txt_embs.cpu().numpy())
            txt_embs = np.concatenate(txt_embs, axis=0)
            np.save(txt_emb_pth, txt_embs) 
        return txt_embs

    def load_fingerprint(self, img_embs, txt_embs):
        txt_logits = txt_embs @ img_embs.t()
        # probs = probs.softmax(dim=-1)
        return txt_logits.cpu().detach().numpy()




