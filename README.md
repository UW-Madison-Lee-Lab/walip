# Unsupervised Word Translation with Clip-based Embedding

## Instructions
---
### Providing a test dictionary
To run the code, you must download or provide the dictionary you want to evaluate on. We provide an example in the [dicts/texts/wiki](dicts/texts/wiki) folder. The [test dictionary](dicts/texts/wiki/wiki_en_fr_test.txt) was obtained [here](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries), which also contains over 110 other bilingual ground truth dictionaries. To prepare the dictionary for evaluation:

* Download the dictionary and place it in the [dicts/texts/wiki](dicts/texts/wiki) folder
* Rename the downloaded dictionary to the format orig_wiki_{src_language}_{tgt_language}_test.txt, replacing the {} with source and target language abbreviations. For example, for the english to french dictionary, it should be named wiki_en_fr_test.txt
* Run the [preprocess.py script](dicts/texts/wiki/preprocess.py) script in the dicts/texts/wiki folder:
```
python preprocess.py <src_lang> <tgt_lang>
```

---
### Fasttext and HTW embeddings
Fasttext embeddings can be downloaded as follows:
```
# English fastText Wikipedia embeddings
curl -Lo data/wiki.en.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
# French fastText Wikipedia embeddings
curl -Lo data/wiki.fr.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
```

HTW word2vec files can be downloaded from [here](https://github.com/gsig/visual-grounding/tree/master/word_vectors)

To prepare fasttext or htw embeddings, you can put them in the datasets/wiki folder for fasttext, and the datasets/htw folder for htw. You can then rename the file to match the convention {data}.{lang}.vec, where data is either "wiki" or "htw" and "lang" is the language abbreviation.

---
### Download the image data
To download the imagenet data, start by downloading the [ILSVRC2012_img_val.tar](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) (about 6.3 GB) from their website, and place it in the datasets/imagenet folder. You may then run these commands in a shell (in the datasets/imagenet folder) to convert it to the ImageFolder format pytorch expects:
```
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
``` 
---
### Obtain CLIP models for source and target languages

We have a [finetuning](src/finetune_clip.py) script to finetune CLIP models, although there are many publicly available clip models we use as well for [english](https://github.com/openai/CLIP), [russian](https://github.com/ai-forever/ru-clip), [japanese](https://github.com/rinnakk/japanese-clip) and [korean](https://github.com/jaketae/koclip). You can follow instructions on their github to run the pip install command.

To use your own finetuned CLIP model, put the checkpoint into the results/clips folder with the name best_{lang}.pt where lang is the language abbreviation. 

---
### Running Scripts
To run translation and evaluation, you can use the [run_translation](src/scripts/run_translation.sh) script. Ensure all the configuration settings match. In particular:
* configs/cuwt.json: configs for different working modes
* configs/langs.json: configs for each language
* configs/settings.json: generic configs

Then you can run in the src folder:

```
./scripts/run_translation.sh <src_lang> <tgt_lang>
```
The best recall@1 and recall@10 will be printed at the end, and the best mapping will be saved into the results/mapping/ours folder