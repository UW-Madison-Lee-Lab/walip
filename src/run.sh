emb=fp
map=nn
set=val
sim=csls
python main.py -e $emb -t $set -i cifar100 -w cifar100 -m $map -s $sim
