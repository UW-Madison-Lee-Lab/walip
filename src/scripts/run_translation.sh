emb=fp
map=no
set=val
sim=ranking
python main.py -e $emb -t $set -i imagenet -w cifar100 -m $map -s $sim --work_mode translation -large_model --num_prompts 1 --tgt_lang sw  -using_filtered_images -reuse_text_embedding -reuse_image_embedding  #--src_lang sw #-reuse_image_data #-reuse_embedding 
# -using_filtered_images #-reuse_embedding 
#-reuse_text_embedding -supervised 
#  python main.py -e $emb -t $set -i imagenet -w imagenet -m $map -s $sim -analysis

# python main.py  -i coco -w coco -analysis