

lang=ko
cd texts/wiki/
cp wiki_en_"$lang"_"$lang"_test.txt wiki_en_"$lang"2_"$lang"2_test.txt
cp wiki_en_"$lang"_en_test.txt wiki_en_"$lang"2_en_test.txt
cp wiki_en_"$lang"_test.txt wiki_en_"$lang"2_test.txt
cd ../../

cd embeddings/

cd fasttext/wiki/
cp fasttext_wiki_en_"$lang"_en_test.npy fasttext_wiki_en_"$lang"2_en_test.npy
cp fasttext_wiki_en_"$lang"_"$lang"_test.npy fasttext_wiki_en_"$lang"2_"$lang"2_test.npy
cd ../../

cd fp/imagenet_s_wiki
cp fp_imagenet_s_wiki_en_"$lang"_en_test_k3.npy fp_imagenet_s_wiki_en_"$lang"2_en_test_k3.npy
cp fp_imagenet_s_wiki_en_"$lang"_"$lang"_test_k3.npy fp_imagenet_s_wiki_en_"$lang"2_"$lang"2_test_k3.npy
cd ../../