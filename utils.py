# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:08:54 2019

@author: Guilherme
"""
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import pandas as pd

from nltk.tree import Tree
import numpy as np
import Levenshtein as lev
import classes

class Bot:

    def __init__(self,data,nicknames,nlp,context,name='Alfie'):
        self.name = name
        self.data = data
        self.nlp = nlp
        self.current_relation = None
        self.current_character = None
        self.character_dict = self.load_characters(data,nicknames)
        self.context = context
        self.load_context()
    
    
    ##USER INTERFACE SECTION
    def run(self):
        ''' Run the bot '''
        print('Press q or quit to exit\n')
        print('Hi! My name is ' + self.name + ' how may I help you?')
        
        while True:
            question = input("Write your question:\n\n")
            if question == 'q' or question == 'quit':
                break
            parsed = parse_sentences(question,self.nlp)
            answer= self.answer_question(parsed)

            if answer[1] == 'character':
                answer = answer[0]
                if answer == 'Error':
                    print('No character found.')
                else:
                    #If there are multiple characters
                    verb = ''
                    s_a = answer.split(', ')
                    l_s_a = len(s_a)
                    if l_s_a > 1:
                        verb = 'are'
                        answer = ', '.join([s_a[i] for i in range(l_s_a -1)])
                        answer = ' and '.join([answer,s_a[-1]])
                    else:
                        verb = 'is'
                    print(answer + ' ' + verb + ' ' + self.current_character.title() + "'s " + self.current_relation+'.')
            else:
                print(answer[0])
    
    def send_message(self,message):
        print(message)
    
    def resolve_conflict(self,options,type_='clarification'):
        ''' Used when query results return multiple choices '''
        string = ''
        #Create strin with options
        for i,option in enumerate(options):
            string = string +str(i+1)+': ' + option.title()+'\n'
        #Get the user to select one
        while True:
            s = ''
            if type_ == 'similarity':
                s = 'Who do you mean by ' + self.current_character.title() + '?'
            elif type_ == 'clarification':
                s = self.current_character.title() + " has several " +self.current_relation+"s."
            choice = input(s+"\nPlease select one by choosing a number from 1 to "+str(len(options)) +":\n\n"+string+"\n")
            #If choice is valid return
            if choice in [str(i) for i in range(1,len(options)+1)]:
                return int(choice)-1
    
    #BACKEND SECTION
    def load_characters(self,data,nicknames):
        unique_characters = list(set(np.concatenate([data['character1'],data['character2']])))
        character_dict = []
        
        for character in unique_characters:
            alias_l = list(nicknames[nicknames['character'] == character].values[:,1])
            character_e = classes.Character(character,alias_l)
            character_dict.append((character,character_e))
            for alias in alias_l:
                character_dict.append((alias,character_e))
        character_dict =dict(character_dict)
        
        #Insert relations in character objects
        for row in data.values:
            char1 = character_dict[row[0]]
            char2 = character_dict[row[1]]
            rel = row[2]
            char1.set_relation(char2,rel)
            character_dict[row[0]] = char1
        
        return character_dict
    
    def load_context(self):
        for character in self.context.keys():
            try:
                
                self.character_dict[character.lower()].set_context(self.context[character])
            
            except KeyError:
                pass
            
    def solve_relation(self,relation_clusters):
        ''' Solve found groups '''
        #Get unique groups(cluster) labels
        clusters = np.unique([x[2] for x in relation_clusters])
        #Sort them in descending order
        clusters = clusters[::-1]
        result = None
        #For each cluster label
        for c in clusters:
            #If we found a result on a previous loop
            if result != None:
                #Split if there are multiple characters
                result = result.split(', ')
                choice = 0
                #If there are multiple characters
                if len(result)>1:
                    #If there is a relation to be solved
                    if 'NN' in [rel[1] for rel in relation_clusters]:
                        choice = self.resolve_conflict(result)
                        #Pick the choice
                        result = result[choice].split(' ')
                        #Prepare the result to be inserted into next relation extraction
                        for word in range(len(result)):
                            result[word] = (result[word],'NNP',c)
                        #Join result into the relation cluster
                        relation_clusters = relation_clusters + result
                    else:
                        return result
                else:
                    #Pick the choice
                    result = result[choice].split(' ')
                    #Prepare the result to be inserted into next relation extraction
                    for word in range(len(result)):
                        result[word] = (result[word],'NNP',c)
                    #Join result into the relation cluster
                    relation_clusters = relation_clusters + result

            #Filter words by cluster
            rels = list(filter(lambda x: x[2] == c,relation_clusters))
            #Extract relation from filtered words
            result = self.extract_relation(rels)
            if result == None:
                return ["ERROR"]
            #Remove filtered words
            relation_clusters = [x for x in relation_clusters if x not in rels]
        return [result]

    def extract_relation(self,relation):
        ''' Extract a relation from the dataframe '''
        #Get nnps from words
        character = ' '.join([x[0].lower() for x in relation if x[1] == 'NNP'])
        #Get nouns from words
        noun = ' '.join([x[0] for x in relation if x[1] == 'NN' or x[1] =='NNS'])
        #If no noun exists return character
        self.current_character = character
        character = self.character_solver(character)
        if noun == '':
            return character.name
        self.current_relation = noun
        #Queries the dataframe
        result = query_data(character,noun,self.data)
        #If found 1 character
        l_result = len(result)
        if l_result == 1:
            result = result[0]
        #If found more then 1 character
        elif l_result > 1:
            result = ', '.join(result)
        else:
            result = None
        return result
    
    def answer_question(self,parsed_questions):
        ''' Answers the question '''
        #It allows multiple questions to be asked at one time
        #For each question
        for sentence in range(len(parsed_questions)):
            #Get the dependency parse of the question
            parse = parsed_questions[sentence]['parse']
            #Get the tree
            tree = get_tree(parse)
            global found
            found = []
            #Search tree return tokens that are related to an entity
            #and attributes a group to them ordered by depth
            #i.e the deeper the entity was found in the tree the higher the label
            #Solve relation will solve the groups from the deepest to the shallowest label
            relation_clusters = search_tree(tree,'NP')
            clusters = [x[2] for x in relation_clusters]
            clusters = np.unique(clusters)

                
            answer = self.solve_relation(relation_clusters)

            if len(clusters) == 1 and clusters[0] == 0 and len(answer) == 1:                
                answer = self.character_dict[answer[0].lower()].context
                return (answer,'context')

            answer = ', '.join(answer)
            return (answer.title(),'character')
        
    def character_solver(self,character):
        #Look for character in dict
        try:
            #If it works
            character = self.character_dict[character]
            
        except KeyError:
            #Does not exist
            #search similar words
            similarity_ratios = [[lev.ratio(c,character),c] for c in self.character_dict.keys()]
            similarity_ratios.sort(reverse = True,key = lambda x:x[0])
            
            #Get top 3
            #If the top word is 2 similar assume:
            if similarity_ratios[0][0] > 0.8:
                character = self.character_dict[similarity_ratios[0][1]]
                self.send_message('I assumed that by ' + self.current_character.title() + ' you ment ' + character.name.title())
            else:
                character = similarity_ratios[0:3]
                #resolve conflict
                choice = self.resolve_conflict(np.array(character)[:,1],'similarity')
                character = self.character_dict[character[choice][1]]
        return character
    

def parse_sentences(sentences,nlp):
    ''' Runs Stanford's parser on the input '''
    
    output = nlp.annotate(sentences, properties={'annotators': 'tokenize, ssplit, pos, depparse, parse, openie','outputFormat': 'json'})
    return output['sentences']

def get_tree(parse):
    ''' Returns a tree from a dependency parse '''
    return Tree.fromstring(parse)
    
def draw_tree(tree):
    ''' Draws a tree... '''
    Tree.draw(tree)

def get_root(tree):
    ''' Gets root node of tree '''
    return tree.label()

def get_children(tree):
    ''' Gets children of tree '''
    return [tree[i] for i in range(len(tree))]



def search_tree(tree,target):
    ''' Search tree for target groups '''
    #This method will search the tree for groups that are labeled with target
    global found
    pointer = 0
    children = get_children(tree)
    for child in children:
        try:
            #If the child's label is target
            if get_root(child) == target:
                #Start labelling words
                deep_search(child,target,pointer,get_root(child))
            else:
                #Else keep looking deeper
                search_tree(child,target)
        except:
            #If the get_root method fails it means we have hit a word 
            #(the deepest you can go on the tree)
            #so start going back
            return found
    #Return the group labeled words
    return found
                
def deep_search(tree,target,pointer,pos_label):
    ''' Does a depth first search on the tree '''
    children = get_children(tree)
    for child in children:
        try:
            if get_root(child) == target:
                #If we hit another target we want to group
                #anything bellow it with into a different group
                pointer = pointer + 1
                #Continue searchuing tree now with new pointer
                deep_search(child,target,pointer,get_root(child))
            else:
                #If not then we are still in the same group
                deep_search(child,target,pointer,get_root(child))
        except:
            #Found word, append to array with pos label and pointer
            found.append((child,pos_label,pointer))
            return
    return



def query_data(char2,type_,data):
    #Clean plural
    if type_[-1] == 's':
        type_ = ''.join([type_[c] for c in range(len(type_)-1)])
    type_ = [type_]
    if type_[0] == 'parent':
        type_[0] = 'father'
        type_.append('mother')
    elif type_[0] == 'sibling':
        type_[0] = 'brother'
        type_.append('sister')
    elif type_[0] == 'grandparent':
        type_[0] = 'grandfather'
        type_.append('grandmother')
    elif type_[0] == 'child' or type_[0] == 'children':
        type_[0] = 'son'
        type_.append('daughter')
    elif type_[0] == 'grandchild' or type_[0] == 'grandchildren':
        type_[0] = 'grandson'
        type_.append('granddaughter')
    for t in range(len(type_)):
        type_[t]= type_[t]+'_of'
    #Query on relation objects?
    if len(type_) > 1:
        query_result = data[(data['character2'] == char2.name) & ((data['type'] == type_[0]) | (data['type'] == type_[1]))]['character1']
    else:
        query_result = data[(data['character2'] == char2.name) & (data['type'] == type_[0])]['character1']
    return query_result.values

def ner_tagger(sent):
    sent = tokenizer(sent)
    sent = pos_tagger(sent)
    sent = tree2conlltags(ne_chunk(sent))
    sent = [(token[0],token[1],token[2]) for token in sent]

    #pattern = 'NP: {<DT>?<JJ>*<NN>}'
    #cp = nltk.RegexpParser(pattern)
    #cs = cp.parse(art_processed)
    
    #iob_tagged = tree2conlltags(cs)
    return sent

def tokenizer(sent):
    sent = nltk.word_tokenize(sent)
    return sent

def removePunct(sent):
    sent = [(token[0],token[1],token[2]) for token in sent if not re.search('[^\w-]',token[0])]
    return sent

def pos_tagger(sent):
    sent = nltk.pos_tag(sent)
    #sent = [(token[0].lower(),token[1]) for token in sent]
    return sent