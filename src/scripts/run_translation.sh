emb=fp
map=nn
set=test
sim=csls
python main.py -e $emb -t $set -i coco -w coco -m $map -s $sim -reuse_embedding 
# -using_filtered_images 
#-reuse_text_embedding -supervised 
#  python main.py -e $emb -t $set -i imagenet -w imagenet -m $map -s $sim -analysis

