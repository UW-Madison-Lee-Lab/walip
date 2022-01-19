emb=fasttext
map=hungarian
set=val
sim=inner_prod
python test.py -e $emb -t $set -i cifar10 -w cifar100 -m $map -s $sim
