import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams 



###-----------Build Unigram Dictionary-----------------###

# Data
sentences=['a b a','b a a b','a a a','b a b b','b b a b','a a a b']

unigrams=[]

for elem in sentences:
    unigrams.extend(elem.split())
    #unigrams.extend(elem)                   # Self - Takes the spaces as unigrams too: Not to be used.
#print(unigrams)								# Self    

from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
    unigram_counts[word]/=unigram_total
 
#print(unigram_counts)




###--------------Build bigram dictionary--------------------###
def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]/=tot_count
     
    return model

# bigram_counts= bigram_model(sentences)
# print(bigram_counts)




###-------------Build trigram dictionary---------------------###
def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]/=tot_count
     
    return model

# trigram_counts= trigram_model(sentences)
# print(trigram_counts)




###---------------Test Scores of each model-------------------###
# test_sentences=['a b a b','b a b a','a b b','b a a a a a b','a a a','b b b b a']
# test_unigram_arr=[]

# print('\nUnigram test probabilities\n')

# for elem in test_sentences:
#     p_val=np.prod([unigram_counts[i] for i in elem.split()])
#     test_unigram_arr.append(p_val)
#     print('The sequence '+elem+' has unigram probablity of '+ str(round(p_val,4)))


# print('\nBigram test probabilities\n')
# test_bigram_arr=[]

# for elem in test_sentences:
#     p_val=1
#     for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
#         p_val*=bigram_counts[w1][w2]
#     print('The sequence '+ elem +' has bigram probablity of '+ str(round(p_val,4)))
    
#     test_bigram_arr.append(p_val)


# test_trigram_arr=[]
# print('\nTrigram test probabilities\n')
# for elem in test_sentences:
#     p_val=1
#     for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
#         try:
#             p_val*=trigram_counts[(w1,w2)][w3]
#         except Exception as e:
#             p_val=0
#             break
#     print('The sequence '+ elem +' has trigram probablity of '+ str(round(p_val,4)))
    
#     test_trigram_arr.append(p_val)





# x_axis=[i for i in range(1,4)]

# y_axis=[np.mean(test_unigram_arr), np.mean(test_bigram_arr), np.mean(test_trigram_arr)]

# plt.scatter(x_axis,y_axis)
# plt.show()


###-------------------Task 1 ------------------###

from nltk.corpus import brown
#print(brown.categories())
#print(np.size(brown.sents()[:40000]))
brown_sents = brown.sents()[:40000]
brown_sents_lower = [[word.lower() for word in element if word.isalpha()]for element in brown_sents]
#print(np.size(brown_sents_lower))
brown_sents_final = []
for sublist in brown_sents_lower:
	for item in sublist:
		brown_sents_final.append(item)
#print(np.size(brown_sents_final))

bigram_counts= bigram_model(brown_sents_final)
print(bigram_counts)
brown_sents_final.sort(key = bigram_counts, reverse = True)
print(brown_sents_final)

# trigram_counts = trigram_model(brown_sents_final)
# print(trigram_counts)