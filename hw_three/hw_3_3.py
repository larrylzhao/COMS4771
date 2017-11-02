# https://rare-technologies.com/word2vec-tutorial/
# https://www.kdnuggets.com/2017/03/email-spam-filtering-an-implementation-with-python-and-scikit-learn.html

import os
from collections import Counter
from nltk.corpus import stopwords
import gensim, logging
from stemming.porter2 import stem
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

# np.set_printoptions(threshold='nan')

train_dir_ham = "train_data/ham"
train_dir_spam = "train_data/spam"

ham = [os.path.join(train_dir_ham,f) for f in os.listdir(train_dir_ham)]
ham_holdout = ham[len(ham)-100:len(ham)]
ham = ham[0:len(ham)-100]

spam = [os.path.join(train_dir_spam,f) for f in os.listdir(train_dir_spam)]
spam_holdout = ham[len(spam)-100:len(spam)]
spam = ham[0:len(spam)-100]

emails = ham + spam
holdout = ham_holdout + spam_holdout

def build_dictionary():

    all_words = []
    for i, email in enumerate(emails):
        if i % 1000 == 1:
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

def dic_from_file():
    dicFile = open('dictionary.txt')
    asdf = dicFile.read()
    asdf = asdf[2:len(asdf)-3]
    asdf = asdf.split('), (')
    dictionary = []
    for line in asdf:
        split = line.split(', ')
        word = split[0]
        word = word[1:len(word)-1]
        count = int(split[1])
        entry = (word, count)
        dictionary.append(entry)
    return dictionary

def extract_features(dictionary, files):

    features_matrix = np.zeros((len(files),3000))
    docID = 0
    for j, file in enumerate(files):
        if j % 100 == 0:
            print j
        f = open(file)
        line = f.read()
        words = line.split()
        for word in words:
            word = stem(word.lower())
            wordID = 0
            for i,d in enumerate(dictionary):
                if d[0] == word:
                    wordID = i
                    features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

def main():

    print len(ham)
    print len(spam)
    print len(emails)
    # print holdout
    # print len(holdout)
    train_labels = np.zeros(len(emails))
    train_labels[len(ham):len(emails)] = 1

    ### build dictionary and feature matrix from scratch ###
    # dictionary = build_dictionary()
    # print dictionary
    # dicFile = open('dictionary.txt', 'w')
    # print>>dicFile, dictionary
    # dicFile.close()

    ### load dictionary and feature matrix from file ###
    dictionary = dic_from_file()
    # print dictionary

    train_matrix = extract_features(dictionary, emails)
    print train_matrix
    np.savetxt('feature_matrix.txt', train_matrix, fmt='%f')

    # train_matrix = np.loadtxt('feature_matrix.txt', dtype=float)
    # print train_matrix

    test_labels = np.zeros(200)
    test_labels[100:200] = 1
    test_matrix = extract_features(dictionary, holdout)

    # model1 = MultinomialNB()
    # model1.fit(train_matrix,train_labels)
    # result1 = model1.predict(test_matrix)
    # print confusion_matrix(test_labels,result1)

    model2 = LinearSVC()
    model2.fit(train_matrix,train_labels)
    result2 = model2.predict(test_matrix)
    print confusion_matrix(test_labels,result2)
    print test_labels
    print result2

main()

