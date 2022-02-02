emb=fp
map=nn
set=test
sim=csls
python main.py -e $emb -t $set -i cifar100 -w cifar100 -m $map -s $sim 
# -using_filtered_images -reuse_embedding 
#-reuse_text_embedding -supervised 
#  python main.py -e $emb -t $set -i imagenet -w imagenet -m $map -s $sim -analysis

