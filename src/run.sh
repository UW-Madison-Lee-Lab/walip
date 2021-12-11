emb=cliptext
map=nn
set=val
sim=cosine
python main.py -e $emb -t $set -i cifar100 -w cifar100 -m $map -s $sim -l supervised
