# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:51:39 2019

@author: Guilherme
"""
import utils
nlp = StanfordCoreNLP('http://localhost:9000')
test = 'Arya Stark is the youngest daughter and third child of Lady Catelyn and Lord Ned Stark. Eddard was the head of House Stark, the Lord Paramount of the North, the Lord of Winterfell, and the Warden of the North to King Robert Baratheon. The North is one of the constituent regions of the Seven Kingdoms and House Stark is one of the Great Houses of the realm. House Stark rules the region from their seat of Winterfell.'

def parse_sentences(sentences):
    ''' Runs Stanford's parser on the input '''
    output = nlp.annotate(sentences, properties={'annotators': 'tokenize, ssplit, pos, ner, depparse, parse, openie','outputFormat': 'json'})
    return output['sentences']
global t
t = []
output = parse_sentences(test)
s1 = output[0]['parse']
tree = utils.get_tree(s1)


utils.draw_tree(tree)


class Group:
    
    def __init__(self,depth,label,parent = None):
        self.depth = depth
        self.elements = None
        self.label = label
        self.parent = parent
    
    def set_elements(self,elements_l):
        for el in elements_l:
            if type(el) == Group:
                el.set_parent(self)
        self.elements = elements_l
    
    def set_parent(self,parent_G):
        self.parent = parent_G

s = []
def new_search(tree,depth = 0,label = None):
    
    children = utils.get_children(tree)
    elements = []
    
    for child in children:
        try:
            label = utils.get_root(child)
            elements.append(new_search(child,depth+1,label))
            
        except AttributeError:
            
            elements.append(child)
            global words
            tuples = (child,label,words)
            words = words + 1
            return tuples
        
    group = Group(depth,label)
    group.set_elements(elements)
    global s
    s.append(group)
    return group

def context_analysis(context):
    output = parse_sentences(context)
    global words
    words = 0
    groups = []
    for sentence in output:
        tree = utils.get_tree(sentence['parse'])
        groups.append(new_search(tree))
        
    ner_taggs = utils.ner_tagger(context)
    #Extract Entities
    entity_extraction(groups,ner_taggs)
    #Analyze Relations
    #Possible relations
    




z = 'Arya Stark is the youngest daughter and third child of Lady Catelyn and Lord Ned Stark.'
ner_taggs = utils.ner_tagger(z)
global words
words = 0
output = parse_sentences(z)
s1 = output[0]['parse']
tree = utils.get_tree(s1)
global s
s = []
new_search(tree)
s.sort(reverse = True,key = lambda x: x.depth)
s = s[0]
for element in s.elements:
    index = element[2]
    ner_tagg = ner_taggs[index]
    if ner_tag 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
def simple_bot(question):
    try:
        messages = np.load('messages.csv.npy')
    except:
        messages = []
        for phrase in output:
            phrase['openie']
            messages = messages + [relation['subject']+' ' + relation['relation'] +' ' + relation['object'] for relation in phrase['openie']]
        np.save('messages.csv',messages)
    
    questions = [question]
    
    try:
        messages = np.load('m_embed.csv.npy')
    except:
        with tf.Session() as session:
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          message_embeddings = session.run(embed(messages))
          questions_embeddings = session.run(embed(questions))
          np.save('m_embed.csv',message_embeddings)
    
    sim = []
    for message in message_embeddings:
        sim.append(cosine_similarity(X = message.reshape(1,-1),Y= questions_embeddings[0].reshape(1,-1)))
    print(messages[np.argmax(sim)])
