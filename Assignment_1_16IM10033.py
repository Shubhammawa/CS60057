import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams 

sentences=['a b a','b a a b','a a a','b a b b','b b a b','a a a b'] # data 

unigrams=[]

for elem in sentences:
    unigrams.extend(elem.split())
   
from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
    unigram_counts[word]/=unigram_total
 
print(unigram_counts)