### 
src=$1
tgt=$2

# extract embs
python extract_embeddings.py -s $src -t $tgt -e htw
# filter nouns
python main.py -s $src -t $tgt -w c -p c
# unsupervised 
python main.py -s $src -t $tgt -w c -p u
# robust procrustes
python main.py -s $src -t $tgt -w c -p t
