#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing neccesary libraries and packages
import nltk
import numpy
import pandas
import spacy
import en_core_web_lg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from itertools import product
from sentence_transformers import SentenceTransformer, util


# In[ ]:


#Input Text
input_text = input("Enter your text ")


# In[ ]:


#Input Summary
input_summary = input("Enter your summary ")


# In[ ]:


#Similarity using spaCy
nlp = spacy.load("en_core_web_lg")

#converting inputs to spaCy processable form
text_nlp =nlp(input_text)
Summary = nlp(input_summary)
              
#Removing the tokens from the input texts
text_no_stop_words = nlp(' '.join([str(t) for t in text_nlp if not t.is_stop]))
summary_no_stop_words = nlp(' '.join([str(t) for t in Summary if not t.is_stop]))

print(text_no_stop_words .similarity(summary_no_stop_words))


# In[ ]:


#Corpora Based approach Similarity


# In[ ]:


str1 = input_text
str2 = input_summary


#Defining stopwords for English Language
stop_words = set(stopwords.words("english"))

##Initialising Lists
filtered_sentence1 = []
filtered_sentence2 = []
lemm_sentence1 = []
lemm_sentence2 = []
sims = []
temp1 = []
temp2 = []
simi = []
final = []
same_sent1 = []
same_sent2 = []
#ps = PorterStemmer()

##Defining WordNet Lematizer for English Language
lemmatizer  =  WordNetLemmatizer()


#Tokenizing and removing the Stopwords (srt1)

for words1 in word_tokenize(str1):
    if words1 not in stop_words:
        if words1.isalnum():
            filtered_sentence1.append(words1)

#Lemmatizing (str2)

for i in filtered_sentence1:
    lemm_sentence1.append(lemmatizer.lemmatize(i))
    



#Tokenizing and removing the Stopwords (str2)

for words2 in word_tokenize(str2):
    if words2 not in stop_words:
        if words2.isalnum():
            filtered_sentence2.append(words2)

#Lemmatizing (str2)

for i in filtered_sentence2:
    lemm_sentence2.append(lemmatizer.lemmatize(i))
    

#Similarity index calculation for each word
for word1 in lemm_sentence1:
    simi =[]
    for word2 in lemm_sentence2:
        sims = []
        syns1 = wordnet.synsets(word1)
        syns2 = wordnet.synsets(word2)
        
        for sense1, sense2 in product(syns1, syns2):
            d = wordnet.wup_similarity(sense1, sense2)
            if d != None:
                sims.append(d)
    
        
        if sims != []:        
           max_sim = max(sims)
           #print(max_sim)
           simi.append(max_sim)
             
    if simi != []:
        max_final = max(simi)
        final.append(max_final)




#Final similarity result

similarity_index = numpy.mean(final)
similarity_index = round(similarity_index , 2)
print("Sentence 1: ",str1)
print("Sentence 2: ",str2)
print("Similarity index value : ", similarity_index)

if similarity_index>0.8:
    print("Similar")
else:
    print("Not Similar")


# In[ ]:


#Textual Similarity using Pytorch and SentenceTransformers
model = SentenceTransformer('stsb-roberta-large')
sentence1 = input_text
sentence2 = input_summary
# encode sentences to get their embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())

