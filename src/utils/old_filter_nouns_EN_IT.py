import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Tutuorial # https://realpython.com/nltk-nlp-python/
#nltk.download('punkt')
#nltk.download("stopwords")
#nltk.download('averaged_perceptron_tagger')

'''
## Tokenize
example_string = "Muad'Dib learned rapidly because his first training was in how to learn. And the first lesson of all was the basic trust that he could learn. It's shocking to find how many people do not believe they can learn, and how many more believe learning to be difficult."
print(sent_tokenize(example_string))
print(word_tokenize(example_string))

## Filtering by stop words 
worf_quote = "Sir, I protest. I am not a merry man!"
words_in_quote = word_tokenize(worf_quote)
print(words_in_quote)
stop_words = set(stopwords.words("english"))
filtered_list = []
for word in words_in_quote:
	if word.casefold() not in stop_words:
		filtered_list.append(word)
print(filtered_list)
filtered_list = [word for word in words_in_quote if word.casefold() not in stop_words]
print(filtered_list)

## Stemming
stemmer = PorterStemmer()
string_for_stemming = """
... The crew of the USS Discovery discovered many discoveries.
... Discovering is what explorers do."""
words = word_tokenize(string_for_stemming)
print(words)
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
'''

## Part of speech
sagan_quote = """
... If you wish to make an apple pie from scratch,
... you must first invent the universe."""
words_in_sagan_quote = word_tokenize(sagan_quote)
pos_tags = nltk.pos_tag(words_in_sagan_quote)
print(pos_tags)

target_tags = ['NN'] #, 'NNP', 'NNPS', 'NNS'] # noun
#target_tags = ['NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'PRP$', 'WP$', 'WRB'] # noun, pronoun
filtered_list = [word for word, tag in pos_tags if tag in target_tags]
print(filtered_list)
'''
for word, tag in pos_tags:
	if tag in target_tags:
		filtered_list.append(word)
'''


# Load data from https://github.com/facebookresearch/MUSE
train_txt = open("en-it.0-5000.txt", "r")
train_data = train_txt.read()
train_list = train_data.split( )
train_txt.close()
en_train_list = train_list[0::2]
it_train_list = train_list[1::2]
print(len(en_train_list))
print(len(it_train_list))
print(en_train_list[-10:])
print(it_train_list[-10:])

test_txt = open("en-it.5000-6500.txt", "r")
test_data = test_txt.read()
test_list = test_data.split( )
test_txt.close()
en_test_list = test_list[0::2]
it_test_list = test_list[1::2]
print(len(en_test_list))
print(len(it_test_list))
print(en_test_list[-10:])
print(it_test_list[-10:])

# Filter only nouns & save it as txt file
'''
train_noun_idx = []
for idx, (word, tag) in enumerate(nltk.pos_tag(en_train_list)):
	if tag in target_tags:
		train_noun_idx.append(idx)
'''

train_noun_idx = [idx for idx, (word, tag) in enumerate(nltk.pos_tag(en_train_list)) if tag in target_tags]
en_train_noun = [en_train_list[idx] for idx in train_noun_idx]
it_train_noun = [it_train_list[idx] for idx in train_noun_idx]
test_noun_idx = [idx for idx, (word, tag) in enumerate(nltk.pos_tag(en_test_list)) if tag in target_tags]
en_test_noun = [en_test_list[idx] for idx in test_noun_idx]
it_test_noun = [it_test_list[idx] for idx in test_noun_idx]

print(len(en_train_noun))
print(len(it_train_noun))
print(en_train_noun[:10])
print(it_train_noun[:10])

print(len(en_test_noun))
print(len(it_test_noun))
print(en_test_noun[:10])
print(it_test_noun[:10])

root = 'Nouns/'
dir = os.path.join(root)
if not os.path.exists(dir):
    os.mkdir(dir)

textfile = open(root+"en_train_noun.txt", "w")
for element in en_train_noun:
    textfile.write(element + "\n")
textfile.close()

textfile = open(root+"it_train_noun.txt", "w")
for element in it_train_noun:
    textfile.write(element + "\n")
textfile.close()

textfile = open(root+"en_test_noun.txt", "w")
for element in en_test_noun:
    textfile.write(element + "\n")
textfile.close()

textfile = open(root+"it_test_noun.txt", "w")
for element in it_test_noun:
    textfile.write(element + "\n")
textfile.close()


## This is old code. NOTE: Do not use this
'''
en_train_noun = [word for word, tag in nltk.pos_tag(en_train_list) if tag in target_tags]
it_train_noun = [word for word, tag in nltk.pos_tag(it_train_list) if tag in target_tags]
en_test_noun = [word for word, tag in nltk.pos_tag(en_test_list) if tag in target_tags]
it_test_noun = [word for word, tag in nltk.pos_tag(it_test_list) if tag in target_tags]
'''
