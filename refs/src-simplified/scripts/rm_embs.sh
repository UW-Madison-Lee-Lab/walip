img=cifar100
k=10
word=cifar100
lang=ru
mode=full

rm ../dicts/embeddings/fp/"$img"_u_"$word"/fp_"$img"_u_"$word"_"$lang"_"$mode".npy
rm ../dicts/embeddings/txt_emb/"$word"/txt_emb_"$word"_"$lang"_"$mode".npy
# rm ../dicts/images/"$img"/img_emb_"$img"_"$lang"_k"$k"_u.npy
