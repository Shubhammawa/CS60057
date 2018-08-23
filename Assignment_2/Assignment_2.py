import numpy as np
import nltk

P= np.array([[0.6, 0.4],[0.5,0.5]])

S= np.array([0.5, 0.5])

O= np.array([[0.3,0.2,0.2,0.3],[0.2,0.3,0.3,0.2]])

state={}
state[0]='L'
state[1]='H'

DNA={}
DNA['A']=0
DNA['C']=1
DNA['G']=2
DNA['T']=3

from itertools import product

import time 
def exhaustive_search(sequence):
    
    M= len(sequence)
    state_len= len(S)
    
    # track the best sequence and its score
    best=(None,float('-inf'))
    
    # basically loop will run for |states|^M 
    for ss in product(range(state_len),repeat=M):
        
        score= S[ss[0]]*O[ss[0],DNA[sequence[0]]]
        
        for i in range(1,M):
            score*= P[ss[i-1],ss[i]]*O[ss[i],DNA[sequence[i]]]
            
        
        #print(','.join([state[k] for k in ss]),score)
    
        if score > best[1]:
            best= (ss,score)
    
    return best

sequences=['GGC','GGCAAGATCAT','GAGAGGAGAGAGAGAGAGA']

import time
for sequence in sequences:
    
    t=time.time()
    best=exhaustive_search(sequence)
    t2=time.time()-t
    
    print('For the sequence '+ sequence+ ' of length '+ str(len(sequence))+' time taken was '+ str(round(t2,3))+'s' )
    print('The sequence '+ ','.join([state[k] for k in best[0]])+ ' gave the best score of '+ str(best[1]))
    print('\n')




from nltk.corpus import treebank
from nltk.corpus import brown

corpus = brown.tagged_sents(tagset='universal')[:-100] 

tag_dict={}
word_dict={}

for sent in corpus:
    for elem in sent:
        w = elem[0]
        tag= elem[1]

        if w not in word_dict:
            word_dict[w]=0

        if tag not in tag_dict:
            tag_dict[tag]=0

        word_dict[w]+=1
        tag_dict[tag]+=1

print(len(word_dict))
print(len(tag_dict))
        
test_data= brown.tagged_sents(tagset='universal')[-10:]

print(len(test_data))

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

train_sents= corpus

def word2features(sent,i):
    word = sent[i][0]
    
    features ={
    'bias': 1.0,
    }
                
    return features

def sent2features(sent):
    return [word2features(sent,i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for i,label in sent]


X_train=[sent2features(s) for s in train_sents]
y_train=[sent2labels(s) for s in train_sents]

X_test=[sent2features(s) for s in test_data]
y_test=[sent2labels(s) for s in test_data]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
labels=list(crf.classes_)

metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=labels)

sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))