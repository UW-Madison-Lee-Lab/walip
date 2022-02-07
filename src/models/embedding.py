import numpy as np
import torch, os
import torch.nn.functional as F
from funcy import chunks
from tqdm import tqdm
from utils.image_loader import load_image_data
from utils.helper import save_images
from models.templates import prompts, generate_texts
from models.ops import load_models
import configs


class ClipEmbedding():
    def __init__(self, emb_type, lang, data_mode, opts):
        self.emb_type = emb_type
        self.lang = lang
        self.data_mode = data_mode
        self.opts = opts
        self.str_filter = 's' if self.opts.using_filtered_images else 'u'
        suffix = ''
        print(self.lang, emb_type, 'embedding!!!')
        if self.emb_type == configs.FINGERPRINT:
            suffix = f'_{self.opts.image_data}_{self.str_filter}'
        self.emb_path = os.path.join(self.opts.emb_dir, f'{self.emb_type}{suffix}_{self.opts.word_data}_{self.lang}_{self.data_mode}.npy')
        self.model = None
    
    def load_clip_model(self):
        model_name = configs.model_names[self.lang]
        self.model, self.logit_scale, self.preprocess = load_models(self.lang, model_name, 'coco', device=self.opts.device, large_model=self.opts.large_model)

    def load_embedding(self, vocabs=None):
        if self.opts.reuse_embedding:
            if os.path.isfile(self.emb_path):
                print('.....', 'Reuse', self.emb_path)
                return np.load(self.emb_path, allow_pickle=True)
            else:
                print("No embedding exists!!!")
        ########### New Embeddings ############
        print('.....', "New embedding", self.emb_path)
        if self.emb_type == configs.FASTTEXT:
            embs = np.load(self.emb_path, allow_pickle=True)
        else:
            self.load_clip_model()
            txt_embs = self.load_clip_txt_emb(vocabs)
            if self.emb_type == configs.FINGERPRINT:
                img_embs = self.load_clip_img_emb()
                embs = self.load_fingerprint(img_embs, torch.from_numpy(txt_embs).to(self.opts.device))
            else:
                embs = txt_embs
            np.save(self.emb_path, embs) 
        return embs

    def load_clip_img_emb(self):
        img_emb_pth = os.path.join(self.opts.img_dir, f'img_emb_{self.opts.image_data}_{self.lang}_k{self.opts.num_images}_{self.str_filter}.npy')
        if self.opts.reuse_image_embedding and os.path.isfile(img_emb_pth):
            print('.....', '.....', "Reuse image embedding", img_emb_pth)
            img_embs = np.load(img_emb_pth, allow_pickle=True)
            img_embs = torch.Tensor(img_embs).to(self.opts.device)
        else:
            if self.model is None:
                self.load_clip_model()
            print('.....', '.....', "New image embedding") 
            ########### New Embeddings ############
            img_pth = os.path.join(self.opts.img_dir, f'img_{self.opts.image_data}_{self.lang}_k{self.opts.num_images}_{self.str_filter}.npy')
            if self.opts.reuse_image_data and os.path.isfile(img_pth): 
                print('.....', '.....', '.....', "Reuse image data") 
                images = np.load(img_pth, allow_pickle=True)
            else:
                print('.....', '.....', '.....', "New image data") 
                images = load_image_data(self.opts.image_data, self.opts.num_images, self.opts.using_filtered_images, self.opts.img_dir, self.preprocess)
                np.save(img_pth, images)
            images = torch.Tensor(images)
            # save image 
            save_images(images, f'../results/base_images_{self.str_filter}.png', nrows=int(images.shape[0]**0.5))
            ### Model embeddings
            list_img_embs = []
            with torch.no_grad():
                for batch_ids in tqdm(chunks(32, range(images.shape[0]))):
                    # batch_imgs = images[batch_ids, ...]
                    # img_embs = self.model.image_encoder(batch_imgs.to(self.opts.device)).float()
                    # img_embs = self.model.image_projection(img_embs)
                    # img_embs = F.normalize(img_embs, dim=-1)
                    img_embs = self.model.encode_image(images[batch_ids, ...].to(self.opts.device))
                    list_img_embs.append(img_embs)    
            img_embs = torch.cat(list_img_embs, dim=0)
            # save embs
            np.save(img_emb_pth, img_embs.cpu().numpy())
        return img_embs

    def load_clip_txt_emb(self, vocabs=None):
        txt_emb_pth = os.path.join(self.opts.emb_dir + f'txt_emb_{self.opts.word_data}_{self.lang}_{self.data_mode}.npy')
        if self.opts.reuse_text_embedding and os.path.isfile(txt_emb_pth):
            print('.....', '.....', "Reuse text embedding", txt_emb_pth)
            txt_embs = np.load(txt_emb_pth, allow_pickle=True)
        else:
            if self.model is None:
                self.load_clip_model()
            print('.....', '.....',  "New text embedding") 
            texts = generate_texts(prompts[self.lang], vocabs, k=self.opts.num_prompts)
            txt_embs = []
            K = len(prompts[self.lang]) if -1 == self.opts.num_prompts else self.opts.num_prompts
            bs = 128  * K
            for batch_texts in tqdm(chunks(bs, texts)):
                with torch.no_grad():
                    # encoded_query = self.tokenizer(batch_texts, padding=True, truncation=True,max_length=200)
                    # batch = {key: torch.tensor(values).to(self.opts.device) for key, values in encoded_query.items()}
                    # batch_txt_embs = self.model.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    # batch_txt_embs = self.model.text_projection(batch_txt_embs).float()
                    # batch_txt_embs = F.normalize(batch_txt_embs, dim=-1)
                    batch_txt_embs = self.model.encode_text(batch_texts)
                    # ensemble
                    batch_size = len(batch_texts) // K
                    batch_txt_embs = batch_txt_embs.view(batch_size, K, batch_txt_embs.shape[-1])
                    batch_txt_embs = batch_txt_embs.mean(dim=1)
                txt_embs.append(batch_txt_embs.cpu().numpy())
            txt_embs = np.concatenate(txt_embs, axis=0)
            np.save(txt_emb_pth, txt_embs) 
        return txt_embs

    def load_fingerprint(self, img_embs, txt_embs):
        img_embs = F.normalize(img_embs, dim=1)
        txt_embs = F.normalize(txt_embs, dim=1)
        txt_logits = txt_embs @ img_embs.t()
        # if self.opts.num_images > 1:
        #     K, D = self.opts.num_images, img_embs.shape[0] // self.opts.num_images
        #     txt_logits = txt_logits.view(-1, D, K)
        #     txt_logits = txt_logits.mean(dim=-1)
        probs = self.logit_scale * txt_logits
        probs = probs.softmax(dim=-1)
        return probs.cpu().detach().numpy()




