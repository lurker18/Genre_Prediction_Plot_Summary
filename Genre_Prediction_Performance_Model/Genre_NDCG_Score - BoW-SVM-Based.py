# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:12:09 2020

@author: Hydra18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import heapq

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

movies = pd.read_excel('Movies3.xlsx', header = 0, encoding = 'utf-8')
#movies.info()
sample = movies.loc[:, ['Title', 'Movie_ID', 'Synopsis', 'Genre1', 'Genre2', 'Genre3']]
train = sample
cols = ['Genre1', 'Genre2', 'Genre3']
train['Genre'] = list(train[cols].apply(lambda x: ','.join(x.dropna()).split(','), axis = 1))
train.drop(['Genre1', 'Genre2', 'Genre3'], axis = 1, inplace = True)
#len(train)
#train
all_genres = sum(train.Genre, [])
#len(set(all_genres))


all_genres = nltk.FreqDist(all_genres) # 5-Genres
all_genres_df = pd.DataFrame({'Genre' : list(all_genres.keys()),
                              'Count' : list(all_genres.values())})


all_genres_df.groupby(by = 'Genre').sum()

#g = all_genres_df.nlargest(columns = "Count", n = 50)
#plt.figure(figsize = (12, 15))
#ax = sns.barplot(data = g, x = "Count",  y= "Genre")
#ax.set(ylabel = 'Count')
#plt.show()

def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)    
    text = ' '.join(text.split())
    text = text.lower()
    return text

train['clean_plot'] = train['Synopsis'].apply(lambda x: clean_text(x))
#train['Synopsis'][0]
#train['clean_plot'][0]



def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    d = words_df.nlargest(columns = "count", n = terms)
    plt.figure(figsize = (12, 15))
    ax = sns.barplot(data = d, x = "count", y = "word")
    ax.set(ylabel = 'Word')
    plt.show()
    
    
#freq_words(train['clean_plot'], 50)
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

train['clean_plot'] = train['clean_plot'].apply(lambda x: remove_stopwords(x))

#freq_words(train['clean_plot'], 50)


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(train['Genre'])

Y = multilabel_binarizer.transform(train['Genre'])



X_train, X_test, y_train, y_test = train_test_split(train['clean_plot'], Y, test_size = 0.2, shuffle = True, random_state = np.random.randint(1000))
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33) # 0.25 x 0.8 = 0.2


X_train.shape
y_train.shape
#X_val.shape
#y_val.shape
X_test.shape
y_test.shape



#################################################################################################
# Train
train_wordfreq = {}
for sentence in X_train:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in train_wordfreq.keys():
            train_wordfreq[token] = 1
        else:
            train_wordfreq[token] += 1

train_most_freq = heapq.nlargest(17000, train_wordfreq, key = train_wordfreq.get)

train_sentence_vectors = []
for sentence in X_train:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in train_most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    train_sentence_vectors.append(sent_vec)
    
train_sentence_vectors = np.asarray(train_sentence_vectors)

#############################################################################################
# Test
test_sentence_vectors = []
for sentence in X_test:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in train_most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    test_sentence_vectors.append(sent_vec)
    
test_sentence_vectors = np.asarray(test_sentence_vectors)


##################################################################################################
def train_classifier(X_train, y_train, X_valid=None, y_valid=None, C=1.0, model='lr'):

    if model=='lr':
        model = LogisticRegression(C=C, penalty='l1', dual=False, solver='liblinear')
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
    
    elif model=='svm':
        model = LinearSVC(C=C, penalty='l2', dual=True, loss='squared_hinge')
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
    
    elif model=='nbayes':
        model = MultinomialNB(alpha=1.0)
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
        
    return model


classifier_tfidf = train_classifier(train_sentence_vectors, y_train, model='svm')


train_y_pred = classifier_tfidf.predict(train_sentence_vectors)
test_y_pred = classifier_tfidf.predict(test_sentence_vectors)


# Inverse of my y_prediction into displayable genres
test_y_pred[0]
list(multilabel_binarizer.inverse_transform(test_y_pred)[0])


# F1-Score Evaluation Score - Micro Vs Macro
# Train
print(round(f1_score(y_train, train_y_pred, average = 'micro'), 3))
print(round(f1_score(y_train, train_y_pred, average = 'macro'), 3))
# Test
print(round(f1_score(y_test, test_y_pred, average = 'micro'), 3))
print(round(f1_score(y_test, test_y_pred, average = 'macro'), 3))

# Accuracy_score = Train Vs Test
print(accuracy_score(y_train, train_y_pred))
print(accuracy_score(y_test, test_y_pred))


def infer_genres(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    sentence_vectors = []
    sentence_tokens = nltk.word_tokenize(q)
    sent_vec = []
    for token in train_most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
    sentence_vectors = np.asarray(sentence_vectors)
    q_pred = classifier_tfidf.predict(sentence_vectors)
    return np.array(multilabel_binarizer.inverse_transform(q_pred))

###########################################################################################
def Genre_NDCG_Score(pred_genres, actual_genres):
    zero = 0
    if len(actual_genres) == 0:
        predict_relevance = zero
        best_relevance = zero
        return predict_relevance, best_relevance

    elif len(actual_genres) == 1:
        best_relevance = [3]
        predict_relevance = []
        try:
            if pred_genres[0] == actual_genres[0]:
                predict_relevance.append(3)
            elif pred_genres[0] == actual_genres[1]:
                predict_relevance.append(2)
            elif pred_genres[0] == actual_genres[2]:
                predict_relevance.append(1)
            else:
                predict_relevance.append(zero)
        
            try:
                if pred_genres[1] == actual_genres[0]:
                    predict_relevance.append(3)
                elif pred_genres[1] == actual_genres[1]:
                    predict_relevance.append(2)
                elif pred_genres[1] == actual_genres[2]:
                    predict_relevance.append(1)
                else:
                    predict_relevance.append(zero)
            
            except IndexError:
                return predict_relevance, best_relevance
            
        except IndexError:
            predict_relevance = [zero]
            return predict_relevance, best_relevance

    elif len(actual_genres) == 2:
        best_relevance = [3, 2]
        predict_relevance = []
        try:
            if pred_genres[0] == actual_genres[0]:
                predict_relevance.append(3)
            elif pred_genres[0] == actual_genres[1]:
                predict_relevance.append(2)
            elif pred_genres[0] == actual_genres[2]:
                predict_relevance.append(1)
            else:
                predict_relevance.append(zero)
        
            try:
                if pred_genres[1] == actual_genres[0]:
                    predict_relevance.append(3)
                elif pred_genres[1] == actual_genres[1]:
                    predict_relevance.append(2)
                elif pred_genres[1] == actual_genres[2]:
                    predict_relevance.append(1)
                else:
                    predict_relevance.append(0)
                return predict_relevance, best_relevance
                    
            except IndexError:
                if len(pred_genres) == 1:
                    return predict_relevance, best_relevance
                else:
                    predict_relevance.append(zero)
                    return predict_relevance, best_relevance
        
        except IndexError:
            if len(pred_genres) == 0:
                predict_relevance.append(zero)
                return predict_relevance, best_relevance
            else:
                zeros = [0, 0]
                predict_relevance = zeros
                return predict_relevance, best_relevance

    elif len(actual_genres) == 3:
        best_relevance = [3,2,1]
        predict_relevance = []
        try:
            if pred_genres[0] == actual_genres[0]:
                predict_relevance.append(3)
            elif pred_genres[0] == actual_genres[1]:
                predict_relevance.append(2)
            elif pred_genres[0] == actual_genres[2]:
                predict_relevance.append(1)
            else:
                predict_relevance.append(zero)
        
            try:
                if pred_genres[1] == actual_genres[0]:
                    predict_relevance.append(3)
                elif pred_genres[1] == actual_genres[1]:
                    predict_relevance.append(2)
                elif pred_genres[1] == actual_genres[2]:
                    predict_relevance.append(1)
                else:
                    predict_relevance.append(zero)
                
            except IndexError:
                return predict_relevance, best_relevance
                
            try:
                if pred_genres[2] == actual_genres[0]:
                    predict_relevance.append(3)
                elif pred_genres[2] == actual_genres[1]:
                    predict_relevance.append(2)
                elif pred_genres[2] == actual_genres[2]:
                    predict_relevance.append(1)
                else:
                    predict_relevance.append(zero)
                    
                return predict_relevance, best_relevance
        
            except IndexError:
                if len(pred_genres) == 1:
                    return predict_relevance, best_relevance
                elif len(pred_genres) == 2:
                    return predict_relevance, best_relevance

        except IndexError:
            if len(pred_genres) == 0 and len(actual_genres) == 0:
                best_relevance = [0]
                predict_relevance = [zero]
                return predict_relevance, best_relevance
            elif len(pred_genres) == 0 and len(actual_genres) == 1:
                best_relevance = [3]
                predict_relevance = [zero]
                return predict_relevance, best_relevance
            elif len(pred_genres) == 0 and len(actual_genres) == 2:
                best_relevance = [3, 2]
                predict_relevance = [zero]
                return predict_relevance, best_relevance
            elif len(pred_genres) == 0 and len(actual_genres) == 3:
                best_relevance = [3,2,1]
                predict_relevance = [zero]
                return predict_relevance, best_relevance


def cumulative_gain_score(rel_list, p):
    return sum(rel_list[:p])
    
def discounted_cumulative_gain_score(rel_list, p):
    dcg = rel_list[0]
    for idx in range(1, p):
        dcg += (rel_list[idx] / np.log2(idx+1))
    return dcg

###############################################################################
# Test Sample
for i in range(2):
    k = X_test.sample(1).index[0]
    print("Movie: ", train['Title'][k], "\nPredicted Genres: ", infer_genres(X_test[k])), print("Actual Genres: ", train['Genre'][k], "\n")

############################################################################
# Train Sample
for i in range(2):
    k = X_train.sample(1).index[0]
    print("Movie: ", train['Title'][k], "\nPredicted Genres: ", infer_genres(X_train[k])), print("Actual Genres: ", train['Genre'][k], "\n")
    
###############################################################################
# Train : NDCG-Score
ndcg = []
for i in range(len(X_train)): # 
    #k = X_train.sample(1).index[0]
    k = X_train.index[i]
    pred_genres = infer_genres(X_train[k]).ravel() # Predicted_Genres
    actual_genres = np.array(train['Genre'][k]) # Actual_Genres
    rank_relevant_score, ideal_vector = Genre_NDCG_Score(pred_genres, actual_genres)

    # CG / ICG / DCG / IDCG    
    #print("CG :", cumulative_gain_score(rank_relevant_score, p=len(rank_relevant_score)))
    #print("ICG :", cumulative_gain_score(ideal_vector, p=len(ideal_vector)))
    #print("DCG :", discounted_cumulative_gain_score(rank_relevant_score, p=len(rank_relevant_score)))
    #print("IDCG :", discounted_cumulative_gain_score(ideal_vector, p=len(ideal_vector)))

    dcg = discounted_cumulative_gain_score(rank_relevant_score, p = len(rank_relevant_score))
    idcg = discounted_cumulative_gain_score(ideal_vector, p = len(ideal_vector))

    # NDCG
    temp = dcg / idcg
    ndcg.append(temp)
    print("NDCG :", np.mean(ndcg))



#len(ndcg) == len(X_train)
Train_Final_NDCG_Score = np.mean(ndcg)
print(round(Train_Final_NDCG_Score, 3))


##########################################################################################
###############################################################################
# Test : NDCG-Score
ndcg = []
for i in range(len(X_test)):
    k = X_test.sample(1).index[0]
    pred_genres = infer_genres(X_test[k]).ravel() # Predicted_Genres
    actual_genres = np.array(train['Genre'][k]) # Actual_Genres
    rank_relevant_score, ideal_vector = Genre_NDCG_Score(pred_genres, actual_genres)

    # CG / ICG / DCG / IDCG    
    #print("CG :", cumulative_gain_score(rank_relevant_score, p=len(rank_relevant_score)))
    #print("ICG :", cumulative_gain_score(ideal_vector, p=len(ideal_vector)))
    #print("DCG :", discounted_cumulative_gain_score(rank_relevant_score, p=len(rank_relevant_score)))
    #print("IDCG :", discounted_cumulative_gain_score(ideal_vector, p=len(ideal_vector)))

    dcg = discounted_cumulative_gain_score(rank_relevant_score, p = len(rank_relevant_score))
    idcg = discounted_cumulative_gain_score(ideal_vector, p = len(ideal_vector))

    # NDCG
    temp = dcg / idcg
    ndcg.append(temp)
    print("NDCG :", np.mean(ndcg))


#len(ndcg) == len(X_test)
Test_Final_NDCG_Score = np.mean(ndcg)
print(round(Test_Final_NDCG_Score, 3))

