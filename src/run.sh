emb=fp
map=hungarian
set=test
sim=csls
python main.py -e $emb -t $set -i cifar100 -w composite -m $map -s $sim
