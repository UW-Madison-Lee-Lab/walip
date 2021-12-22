emb=fp
map=hungarian
set=test
sim=csls
python main.py -e $emb -t $set -i imagenet -w composite -m $map -s $sim
