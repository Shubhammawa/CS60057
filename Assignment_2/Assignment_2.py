#------------------Task 1--------------------#

# Generative models: Model joint probability
# 	1. Naive Bayes Classifier
# 	2. Linear discrimnant analysis
# 	3. Generative Adversial Networks

# Discriminative models: Model conditional probability
# 	1. Neural Networks
# 	2. Logistic Regression
# 	3. SVM
# 	4. Decision Trees

#--------------------Task 2--------------------#

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


###----------------------HMM---------------------###

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
word_dict['UNK'] = 0
print(len(word_dict))
n_words = len(word_dict)	# Number of unique words in the dictionary
#print(word_dict)
print(len(tag_dict))
n_tags = len(tag_dict)		# Number of unique POS tags in the dictionary
#print(tag_dict)        
test_data= brown.tagged_sents(tagset='universal')[-10:]

print(len(test_data))

# Creating dictionary of indices - will be used in creating matrices
word_index = {}
i = 0
for word in word_dict:
	word_index[word] = i
	i+= 1

state_index = {}
index_state = {}
i = 0
for tag in tag_dict:
	state_index[tag] = i
	index_state[i] = tag 
	i+= 1

# Start Matrix
S = np.zeros(n_words)
#Transition Matrix
T = np.zeros([n_words,n_words])
# Emission Matrix
E = np.zeros([n_words,n_words])

for sent in corpus:
	S[state_index[sent[0][1]]] += 1

	w = sent[0][0]
	tag = sent[0][1]
	E[state_index[tag]][word_index[w]] +=1

	ex_tag = tag 

	for i in range(1,len(sent)):
		w = sent[i][0]
		tag = sent[i][1]
		T[state_index[ex_tag]][state_index[tag]] += 1
		E[state_index[tag]][word_index[w]] += 1
S = np.divide(S,S.sum())
#print(S)
#print(np.sum(S))

for i in range(n_words):
	T[i] = T[i]/np.sum(T[i])

k = 0.1 		# Smoothing factor
for i in range(n_words):
    N = np.sum(E[i])
    for j in range(n_words):
        E[i][j] = (E[i][j] + k)/(N + k*n_words)


def viterbi_algo(O, S, Tr, E, word_index):
    N = len(S) # No of states
    T = len(O) # No of words in Observtion O
    path_prob = np.zeros([N, T])
    back_pointer = np.zeros([N, T], dtype=np.int)
    best_path = np.zeros(T, dtype=np.int)

    for i in range(T):
        if O[i] not in word_index:
            O[i] = 'UNK'

    for s in range(N):
        path_prob[s][0] = S[s] * E[s][word_index[O[0]]]
        back_pointer[s][0] = -1 

    for t in range(1, T):
        for s in range(N):
            ems = E[s][word_index[O[t]]]
            arr = np.array([ (path_prob[s1][t-1] * Tr[s1][s] * ems)  for s1 in range(N) ])
            path_prob[s][t] = np.max(arr)
            back_pointer[s][t] = np.argmax(arr)

    arr = np.array([ path_prob[s][T-1] for s in range(N) ])
    best_path_prob = np.max(arr)
    best_path[T-1] = np.argmax(arr)
    for t in range(T-1, 0, -1):
        best_path[t-1] = back_pointer[best_path[t]][t]

    return best_path, best_path_prob

n_test_words = 0
n_corr_pred_test_words = 0

HMM_y_test = []
HMM_y_pred = []

for sent in test_data:
    O = [s[0] for s in sent]
    TT = len(O)
    best_path, best_score=viterbi_algo(O, S, T, E, word_index)
    
    tags = [s[1] for s in sent]
    pred_tags = [index_state[k] for k in best_path]
    HMM_y_test.append(tags)
    HMM_y_pred.append(pred_tags)
    print(len(tags), len(pred_tags), len(HMM_y_test), len(HMM_y_pred))
    
    print('For the sequence "'+ ' '.join(O)+ '" of length '+ str(TT)+' time taken was '+ str(round(t2,3))+'s' )
    print('The sequence '+ ','.join(pred_tags)+ ' gave the best score of '+ str(best_score))
    print('Actual sequence of tags : '+ ','.join(tags))
    print('\n')

    for t in range(TT):
        if tags[t] == pred_tags[t]:
            n_corr_pred_test_words += 1
        n_test_words += 1

accuracy = 100.0*n_corr_pred_test_words/n_test_words
print('Accuracy = ' + str(accuracy))
print("\n\nHMM\n")
print(metrics.flat_classification_report(
    HMM_y_test, HMM_y_pred, digits=3))


# ###---------------------------CRF--------------------------###
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