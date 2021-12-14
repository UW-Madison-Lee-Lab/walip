emb=ganfasttext
map=hungarian
set=test
sim=csls
python main.py -e $emb -t $set -i cifar100 -w cifar100 -m $map -s $sim
