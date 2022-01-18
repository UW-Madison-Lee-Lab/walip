emb=cliptext
map=nn
set=test
sim=csls
python main.py -e $emb -t $set -i cifar10 -w cifar10 -m $map -s $sim
