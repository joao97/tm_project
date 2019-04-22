#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:29:43 2019

@author: joaofernandes
"""

import scrapy
import re 
from scrapy.crawler import CrawlerProcess
import numpy as np 
import json
import pandas as pd

#loading the characters dataset to a dictionary
with open("characters.json") as f:
    data = json.load(f)

characters_name = []
for character in data['characters']:
    characters_name.append(character['characterName'])
    
    
    
#Creating an empty dictionary that will contain the scripts of all the episodes
info = {}
#Extracting the episodes number, season and scripts
class script_spider(scrapy.Spider):
    name = "scripts"

    def start_requests(self):
        for character in characters_name:
            yield scrapy.Request(url='https://gameofthrones.fandom.com/wiki/'+character.replace(' ','_'), callback=self.parse, meta={'character':character})
    def parse(self, response):
        i=1
        text = re.sub('<.*?>','',response.xpath('//*[@id="mw-content-text"]/p[1]').getall()[0]).strip()
        while(text==''):
            i=i+1
            text = re.sub('<.*?>','',response.xpath('//*[@id="mw-content-text"]/p['+str(i)+']').getall()[0]).strip()
        text = text + re.sub('<.*?>','',response.xpath('//*[@id="mw-content-text"]/p['+str(i+1)+']').getall()[0]).strip()
        global info
        info[response.meta.get('character')] = text
#Running spider        
process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})
process.crawl(script_spider)
process.start(True)

np.save('context_dict',info)

import nltk
from itertools import chain
from nltk.corpus import wordnet
from nltk.corpus import stopwords

#normalization
def normalize(tokens):
    #remove punctuation tokens and special characters
    tokens = [re.sub(r'[^\w\s]|_','',token).lower() for token in tokens if not re.match('^\.{3}|[^\w\s\d]$',token)]
    #removing stopwords
    stop_words=set(stopwords.words("english"))
    filtered_sent=[]
    for w in tokens:
        if w!='' and w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

#method to handle with the different types of words, such as, names, verbs or adjectives
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(token)) for token in tokens]
    return tokens

#info['query'] = "who is cersei lannister's mother?"


terms = []
corpus = []
sentences = []
raw_sentences = []
for character in info.keys():
    character_tokens = nltk.sent_tokenize(info[character])
    #character_tokens = list(chain(*character_tokens))
    #character_tokens = normalize(character_tokens)
    #character_tokens = lemmatize(character_tokens)
    #terms.append(character_tokens)
    #corpus.append(' '.join(character_tokens))
    [raw_sentences.append(re.sub(r'\b([H|h|Sh|s]e)\b',character,sentence)) for sentence in character_tokens]
    [sentences.append([character, sentence]) for sentence in character_tokens]

sentences = pd.DataFrame(sentences, columns = ['character','sentence'])


from sklearn.feature_extraction.text import TfidfVectorizer


#calculating cosine similarity with sklearn package
from sklearn.metrics.pairwise import cosine_similarity



import string
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def response(user_response, sentences):
    robo_response=''
    sentences.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentences)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sentences[idx]
        return robo_response
    

user_response = "daenerys brother?"
sentences = raw_sentences.copy()


robo_response=''
sentences.append(user_response)
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
tfidf = TfidfVec.fit_transform(sentences)
vals = cosine_similarity(tfidf[-1], tfidf)
idx=vals.argsort()[0][-2]
flat = vals.flatten()
flat.sort()
req_tfidf = flat[-2]
if(req_tfidf==0):
    robo_response=robo_response+"I am sorry! I don't understand you"
else:
    robo_response = robo_response+sentences[idx]
print(robo_response)
   



'uncle': ['brother ']
