# -*- coding: utf-8 -*-

from textblob import TextBlob
import re
import torch
import nltk
nltk.download('punkt')
import numpy as np
import utils
from pycorenlp import StanfordCoreNLP
import pandas as pd
import classes
import pickle
import logging

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
nlp = StanfordCoreNLP('http://localhost:9000')

#Load Data

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
nicknames = pd.read_csv('nicknames.csv')
nicknames = nicknames.set_index('Unnamed: 0')

data = pd.read_csv('relationships.csv')
data = data.set_index('Unnamed: 0')
data.loc[data['type'] == 'cousinof']  = 'cousin_of'
data = data.drop_duplicates()
context_info = load_obj('context_dict')

bot = utils.Bot(data,nicknames,nlp,context_info)
bot.run()


