# https://rare-technologies.com/word2vec-tutorial/

# import modules & set up logging
import os
from collections import Counter
from nltk.corpus import stopwords
import gensim, logging
from stemming.porter2 import stem

train_dir_ham = "train_data/ham"
train_dir_spam = "train_data/spam"

def build_dictionary(dir):
    emails = [os.path.join(dir,f) for f in os.listdir(dir)]
    all_words = []
    for i, email in enumerate(emails):
        print i
        with open(email) as m:
            for line in m:
                words = line.split()
                for word in words:
                    word = stem(word.lower())
                all_words += words
    dictionary = Counter(all_words)
    stopWords = set(stopwords.words('english'))
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in stopWords:
            del dictionary[item]

    dictionary = dictionary.most_common(3000)
    return dictionary

def main():

    dictionary = build_dictionary(train_dir_ham)
    print dictionary

main()